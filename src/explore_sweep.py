"""FREE exploration sweep — the online fine-tune arm collapsed to one move (98% usage, entropy 0.10,
reward = single) by epoch 3 because entropy_beta=0.02 couldn't hold the policy open. Before spending
more GPU, sweep entropy_beta x lr on the cached simulator (with m=1 noise) to find settings that reach
the exact-policy ceiling (~+0.38) WITHOUT collapsing to a single arm. Reports, per setting: final eval
ΔRM, max move-usage (collapse metric; want < ~0.7), and final policy entropy. Picks the online config.

Linear head ~ the frozen arm; MLP head ~ the higher-capacity fine-tune arm (collapses more eagerly),
so a beta that keeps the MLP open is the safe online floor. CPU, seconds.

    python src/explore_sweep.py --tag large_7b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from router_bandit import _pca, _R_from_M, _realized, _head

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def run(Ztr, Rtr, Zev, Rev, hp, seed, noise, epochs):
    """Sampled REINFORCE (norm-adv + value baseline + entropy). Returns final eval ΔRM, max-usage, entropy."""
    torch.manual_seed(seed)
    din, K1 = Ztr.shape[1], Rtr.shape[1]
    pol = _head(din, K1, hp["hidden"], 0.0); val = nn.Linear(din, 1)
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=hp["lr"], weight_decay=0.01)
    Zt, Rt, Zv = (torch.tensor(Ztr, dtype=torch.float32), torch.tensor(Rtr, dtype=torch.float32),
                  torch.tensor(Zev, dtype=torch.float32))
    n = len(Ztr); g = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed + 917)
    last_usage = np.ones(K1) / K1; last_ent = np.log(K1)
    for _ in range(epochs):
        pol.train(); val.train(); usage = np.zeros(K1); ents = []
        order = torch.randperm(n, generator=g)
        for s in range(0, n, hp["batch"]):
            idx = order[s:s + hp["batch"]]; z, r_all = Zt[idx], Rt[idx]
            dist = torch.distributions.Categorical(logits=pol(z))
            a = dist.sample()
            r = r_all.gather(1, a[:, None]).squeeze(1).clone()
            if noise > 0:
                r = r + torch.tensor(rng.standard_normal(len(r)) * noise, dtype=torch.float32) * (a > 0).float()
            v = val(z).squeeze(1); adv = (r - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            loss = -(adv * dist.log_prob(a)).mean() - hp["beta"] * dist.entropy().mean() + 0.5 * F.mse_loss(v, r)
            opt.zero_grad(); loss.backward(); opt.step()
            for aa in a.tolist():
                usage[aa] += 1
            ents.append(float(dist.entropy().mean().detach()))
        last_usage = usage / usage.sum(); last_ent = float(np.mean(ents))
    pol.eval()
    with torch.no_grad():
        ev = _realized(pol(Zv).argmax(1).numpy(), Rev).mean()
    return float(ev), float(last_usage.max()), last_ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=1.0)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    ec = np.load(OUT / "enc_embed.npz", allow_pickle=True); H = ec["Htr"]
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]; K = len(S)
    idx_all = np.where(~np.isnan(Msel).all(1))[0]
    single = np.mean([np.nan_to_num(Msel[i, 0], nan=0.0) for i in idx_all])

    betas = [0.02, 0.05, 0.1, 0.2, 0.4]; lrs = [0.02, 0.05]
    heads = [("linear~frozen", 0), ("mlp~finetune", 64)]
    rows = [f"# Exploration sweep (offline, free) — {args.tag}: entropy_beta x lr to STOP collapse\n",
            f"Sampled REINFORCE on cached enc features + swings, m=1 noise σ={args.noise}, {args.seeds} seeds, "
            f"{args.epochs} epochs (matches online). single move {single:+.3f}; exact-policy ceiling ≈ +0.38. "
            f"Collapse = max move-usage; want eval near ceiling AND max-usage < ~0.7 (still conditioning).\n",
            "| head | beta | lr | eval ΔRM | max-usage | entropy | note |", "|---|---|---|---|---|---|---|"]
    best = None
    for hname, hid in heads:
        for beta in betas:
            for lr in lrs:
                evs, us, en = [], [], []
                for seed in range(args.seeds):
                    idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
                    nsp = len(idx); a2 = int(0.8 * nsp)
                    tr, ev = idx[:a2], idx[a2:]
                    Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], args.n_pca)
                    e, u, ent = run(Ztr, _R_from_M(Msel[tr]), Zev, _R_from_M(Msel[ev]),
                                    dict(hidden=hid, lr=lr, beta=beta, batch=32), seed, args.noise, args.epochs)
                    evs.append(e); us.append(u); en.append(ent)
                emean, umean, enmean = np.mean(evs), np.mean(us), np.mean(en)
                collapsed = umean > 0.7
                note = "COLLAPSE" if collapsed else ("ok" if emean > single + 0.03 else "weak")
                rows.append(f"| {hname} | {beta} | {lr} | {emean:+.3f} | {umean:.2f} | {enmean:.2f} | {note} |")
                # pick: MLP head (the collapse-prone one), not collapsed, max eval
                if hname == "mlp~finetune" and not collapsed:
                    if best is None or emean > best[0]:
                        best = (emean, beta, lr, umean)
    rows += ["", "## Pick (MLP≈fine-tune arm, not collapsed, best eval)"]
    if best:
        rows.append(f"- **entropy_beta={best[1]}, lr={best[2]}** → eval {best[0]:+.3f}, max-usage {best[3]:.2f} "
                    "⇒ use these online (and lower lr_enc so the encoder can't drive collapse either).")
    else:
        rows.append("- No non-collapsed MLP setting beat single by >0.03 within these betas ⇒ raise beta further, "
                    "lower lr, and/or add entropy annealing; also consider that online collapse-to-single may be "
                    "the honest ceiling (weak conditioning) rather than an exploration artifact.")
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_explore_sweep_{args.tag}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
