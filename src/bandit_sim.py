"""B0 — de-risk the online bandit BEFORE spending GPU, using the cached swing matrix as a free
online-bandit SIMULATOR. Two jobs, both offline (rewards read from swing_train.npz, no generation):

  (1) MEASURE E (sample efficiency). Run the online loop we'd run on GPU — each step presents a
      train prompt, the policy samples one arm, we "generate" by looking up its reward — and track
      held-out eval ΔRM vs generations consumed. E = generations-per-prompt (epochs) to reach a
      target (fraction of the full-information exact-policy level). This converts the E ESTIMATE in
      the B cost plan into a MEASURED number. Online pulls are m=1, so we inject fresh per-pull RM
      noise (sigma) — sigma=0 is the optimistic (as-if-averaged) bound, sigma~1.0 the realistic m=1
      case — to bracket E honestly.

  (2) HARDEN REINFORCE (fix rung-2's variance loss to exact-policy). Adds advantage normalization +
      value baseline + larger batch, and re-tests whether the hardened online update recovers the
      exact-expectation policy. Gate for going online: hardened ≈ exact, and E small (< ~15).

    python src/bandit_sim.py --tag large_7b --rep enc --seeds 12
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from router_bandit import _pca, _boot, _head, _realized, _R_from_M, fit_policy_exact

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"
GEN_S = 2.84   # MEASURED s/generation (large_7b shard: 28150 s / 9940 gens, A5000, 768-tok cap)


def online_sim(Ztr, Rtr, Zev, Rev, hp, seed, noise, max_epochs):
    """One online run. Returns per-epoch eval ΔRM (argmax policy). Each epoch = len(Ztr) generations."""
    torch.manual_seed(seed)
    din, K1 = Ztr.shape[1], Rtr.shape[1]
    pol = _head(din, K1, hp["hidden"], hp["dropout"])
    val = nn.Linear(din, 1)
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=hp["lr"], weight_decay=hp["wd"])
    Zt, Rt, Zv = torch.tensor(Ztr, dtype=torch.float32), torch.tensor(Rtr, dtype=torch.float32), torch.tensor(Zev, dtype=torch.float32)
    n = len(Ztr)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed + 917)
    curve = []
    for _ in range(max_epochs):
        pol.train(); val.train()
        order = torch.randperm(n, generator=g)
        for s in range(0, n, hp["batch"]):
            idx = order[s:s + hp["batch"]]
            z, r_all = Zt[idx], Rt[idx]
            dist = torch.distributions.Categorical(logits=pol(z))
            a = dist.sample()                                        # one arm/prompt = the "1 generation"
            r = r_all.gather(1, a[:, None]).squeeze(1).clone()
            if noise > 0:                                            # fresh m=1 RM noise on move arms (decline=0 exactly)
                eps = torch.tensor(rng.standard_normal(len(r)) * noise, dtype=torch.float32)
                r = r + eps * (a > 0).float()
            v = val(z).squeeze(1)
            adv = (r - v).detach()
            if hp["norm_adv"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            loss = -(adv * dist.log_prob(a)).mean() - hp["beta"] * dist.entropy().mean() + hp["cv"] * F.mse_loss(v, r)
            opt.zero_grad(); loss.backward(); opt.step()
        pol.eval()
        with torch.no_grad():
            curve.append(_realized(pol(Zv).argmax(1).numpy(), Rev).mean())
    return np.array(curve)


def _first_reach(curve, target):
    hit = np.where(curve >= target)[0]
    return int(hit[0] + 1) if len(hit) else None                    # epochs (=gens/prompt); None if never


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--rep", default="enc", choices=["last", "mean", "enc"])
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--max_epochs", type=int, default=40)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    t0 = time.time()

    if args.rep == "enc":
        ec = np.load(OUT / "enc_embed.npz", allow_pickle=True); H = ec["Htr"]; rep_name = str(ec["encoder"])
    else:
        st = np.load(OUT / "states.npz", allow_pickle=True); L = int(st["layers"][-1])
        H = st[f"Htr_{L}_{args.rep}"]; rep_name = f"LLM layer {L} {args.rep}"
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]; K = len(S)
    idx_all = np.where(~np.isnan(Msel).all(1))[0]

    hp = dict(hidden=0, dropout=0.0, lr=0.02, wd=0.01, beta=0.02, cv=0.5, batch=128, norm_adv=True)
    ex_grid = [dict(hidden=h, dropout=0.3, lr=0.05, wd=wd, beta=0.0, epochs=800, patience=40)
               for h in (0, 64) for wd in (0.01, 0.1)]
    noises = [0.0, 1.0]
    epochs_report = [1, 2, 3, 5, 8, 12, 20, 30, 40]

    curves = {s: [] for s in noises}                                 # per-noise list over seeds of eval curves
    exact_ref, single_ref, hardfinal = [], [], []
    for seed in range(args.seeds):
        idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
        n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
        tr, va, ev = idx[:a], idx[a:b], idx[b:]
        Ztr, Zva, Zev = _pca(H[tr], [H[tr], H[va], H[ev]], args.n_pca)
        Rtr, Rva, Rev = _R_from_M(Msel[tr]), _R_from_M(Msel[va]), _R_from_M(Msel[ev])
        single_ref.append(np.nan_to_num(Msel[ev, 0], nan=0.0).mean())
        # full-information reference (best offline policy), val-selected
        best_v, best_ev = -1e18, None
        for hpe in ex_grid:
            act = fit_policy_exact(Ztr, Rtr, Zva, Rva, hpe, "cpu", seed)
            vr = _realized(act(Zva), Rva).mean()
            if vr > best_v: best_v, best_ev = vr, _realized(act(Zev), Rev).mean()
        exact_ref.append(best_ev)
        for s in noises:
            c = online_sim(Ztr, Rtr, Zev, Rev, hp, seed, s, args.max_epochs)
            curves[s].append(c)
        hardfinal.append(curves[0.0][-1][-1])                        # noiseless hardened final = rung-2 re-test

    exact_ref = np.array(exact_ref); single_ref = np.array(single_ref); hardfinal = np.array(hardfinal)
    d_hard_ex = hardfinal - exact_ref; hlo, hhi = _boot(d_hard_ex)
    tgt = 0.90                                                       # reach 90% of the (single->exact) gain
    dt = time.time() - t0

    rows = [f"# B0 bandit simulator (offline, free) — {args.tag}, rep={args.rep} ({rep_name})\n",
            f"Cached swing log as an online-bandit simulator ({len(idx_all)} prompts, K={K}+decline), PCA-{args.n_pca}, "
            f"{args.seeds} seeds, hardened REINFORCE (norm-adv + value baseline + batch {hp['batch']}). "
            f"Offline CPU {dt:.0f}s, 0 GPU-h, 0 generations. Cost translation at MEASURED {GEN_S:.2f}s/gen.\n",
            f"Reference (mean over seeds): single {single_ref.mean():+.3f} ; full-info exact-policy {exact_ref.mean():+.3f}.\n",
            "## (2) Hardening gate — hardened online REINFORCE vs exact-policy (noiseless, paired)",
            f"- hardened final {hardfinal.mean():+.3f} ; exact {exact_ref.mean():+.3f} ; "
            f"paired {d_hard_ex.mean():+.3f} [{hlo:+.3f}, {hhi:+.3f}] "
            + ("⇒ **recovers exact ⇒ online update rule is sound.**" if d_hard_ex.mean() >= -0.01 else "⇒ **still below exact — variance not yet controlled.**"),
            "", "## (1) Sample-efficiency curve — eval ΔRM vs epochs (=generations/prompt), mean over seeds",
            "| epochs (=gens/prompt) | " + " | ".join(f"σ={s:g}" for s in noises) + " |",
            "|---|" + "---|" * len(noises)]
    for e in epochs_report:
        cells = []
        for s in noises:
            vals = [c[min(e, len(c)) - 1] for c in curves[s]]
            cells.append(f"{np.mean(vals):+.3f}")
        rows.append(f"| {e} | " + " | ".join(cells) + " |")

    rows += ["", "## Measured E (epochs to 90% of the single→exact gain) and implied training cost"]
    for s in noises:
        Es = []
        for i in range(args.seeds):
            target = single_ref[i] + tgt * (exact_ref[i] - single_ref[i])
            e = _first_reach(curves[s][i], target)
            if e is not None: Es.append(e)
        if Es:
            Emean = float(np.mean(Es)); reach = len(Es) / args.seeds
            gph = lambda N: (N * Emean * GEN_S) / 3600.0
            rows.append(f"- σ={s:g}: **E ≈ {Emean:.1f} gens/prompt** ({reach*100:.0f}% of seeds reached target) "
                        f"⇒ training ≈ {gph(1000):.1f} GPU-h @N=1000, {gph(3000):.1f} GPU-h @N=3000.")
        else:
            rows.append(f"- σ={s:g}: target not reached within {args.max_epochs} epochs ⇒ E > {args.max_epochs}; investigate before GPU.")

    rows += ["", "## Reading",
             "- **Gate = (hardened ≈ exact) AND (E small at σ=1.0).** If both hold, B1 online is de-risked at the costed budget.",
             "- σ=1.0 is the realistic m=1 online case; σ=0 is the optimistic (averaged-reward) bound. Read E off σ=1.0.",
             "- All rewards are m_swing point estimates ⇒ E is a planning figure; B1 logs the real generations-to-target."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_bandit_sim_{args.tag}_{args.rep}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
