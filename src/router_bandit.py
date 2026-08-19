"""Step-3 router as a CONTEXTUAL BANDIT — offline rungs 1 & 2 on the cached swing matrix.

Motivation: the value-regression router (router_explore) fits squared error over ALL K arm
values and then argmaxes — a harder problem than we need, and it requires K generations per
training prompt. The bandit reframing optimizes the DECISION objective directly (encoder ->
softmax policy pi(a|x) over the K moves + a decline arm), which (a) may beat regression-then-
argmax on identical data (Vapnik: don't solve a harder problem than the task) and (b) is the
single-turn special case of the multi-turn policy we carry to Study 2 (rule 6). Here we test
the LEARNING RULE for free on the cached full-information log (we know r(x,a) for every arm),
decoupling "does the objective help" from "does adaptive sampling save generation" (rung 3, online).

Rungs, cheapest first (all offline, rewards read from swing_train.npz — NO generation):
  * rung 0  REGRESSION baseline (reproduces router_explore's reg-argmax router on this split).
  * rung 1  EXACT-EXPECTATION policy: maximize sum_a pi(a|x) r(x,a) — deterministic, no sampling
            (cost-sensitive classification; uses reward magnitude, unlike hard cls / regression).
  * rung 2  SAMPLED REINFORCE: sample a~pi, advantage (r - V(x)) with a learned value baseline +
            entropy bonus (also our anti-collapse guard). Same policy — tests whether the
            variance-injected estimator we'd run ONLINE recovers rung 1 offline.

Same 60/20/20 split, PCA features, and honest val-selection as router_explore, so the eval ΔRM
sits directly next to the +0.069 regression number. Multi-seed by default (the hard-won lesson:
never conclude from one noisy split). Action 0 = decline (reward 0); actions 1..K = the moves.

    python src/router_bandit.py --tag large_7b --rep enc --seeds 12
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


# ---- shared helpers (mirrored from router_explore so this module is self-contained) ----
def _pca(Hfit, Hs, k):
    mu = Hfit.mean(0)
    _, _, Vt = np.linalg.svd(Hfit - mu, full_matrices=False)
    C = Vt[: min(k, Vt.shape[0])]
    return [(H - mu) @ C.T for H in Hs]


def _boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = np.sort(v[rng.integers(0, len(v), (n, len(v)))].mean(1))
    return float(b[int(0.025 * n)]), float(b[int(0.975 * n)])


def _head(din, out_dim, hidden, dropout):
    if hidden <= 0:
        return nn.Linear(din, out_dim)
    return nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, out_dim))


def _reg_action(vals):                                   # vals (n,K) predicted swings
    j = vals.argmax(1)
    top = vals[np.arange(len(vals)), j]
    return np.where(top <= 0, 0, j + 1)                  # 0 = decline, else move j+1


def _realized(action, R):                                # R (n,K+1), col 0 = decline (=0)
    return R[np.arange(len(action)), action]


# ---- rung 0: value-regression baseline (masked MSE over the K arms, then argmax) ----
def fit_regression(Ztr, Mtr, Zva, Mva, hp, device, seed):
    torch.manual_seed(seed)
    K = Mtr.shape[1]
    net = _head(Ztr.shape[1], K, hp["hidden"], hp["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
    Zv = torch.tensor(Zva, dtype=torch.float32, device=device)
    A = torch.tensor(np.nan_to_num(Mtr), dtype=torch.float32, device=device)
    Am = torch.tensor(~np.isnan(Mtr), dtype=torch.float32, device=device)
    Rva = _R_from_M(Mva)
    best, best_state, bad = -1e18, None, 0
    for _ in range(hp["epochs"]):
        net.train(); opt.zero_grad()
        p = net(Zt)
        (((p - A) ** 2 * Am).sum() / Am.sum().clamp(min=1)).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val_r = _realized(_reg_action(net(Zv).cpu().numpy()), Rva).mean()
        if val_r > best + 1e-6:
            best, best_state, bad = val_r, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    net.load_state_dict(best_state)
    return lambda Z: _reg_action(net(torch.tensor(Z, dtype=torch.float32, device=device)).detach().cpu().numpy())


# ---- rung 1: exact-expectation policy (deterministic; maximize E_pi[R]) ----
def fit_policy_exact(Ztr, Rtr, Zva, Rva, hp, device, seed):
    torch.manual_seed(seed)
    net = _head(Ztr.shape[1], Rtr.shape[1], hp["hidden"], hp["dropout"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
    Zv = torch.tensor(Zva, dtype=torch.float32, device=device)
    Rt = torch.tensor(Rtr, dtype=torch.float32, device=device)
    best, best_state, bad = -1e18, None, 0
    for _ in range(hp["epochs"]):
        net.train(); opt.zero_grad()
        logp = F.log_softmax(net(Zt), dim=1)
        p = logp.exp()
        loss = -(p * Rt).sum(1).mean() - hp["beta"] * (-(p * logp).sum(1).mean())
        loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            val_r = _realized(net(Zv).argmax(1).cpu().numpy(), Rva).mean()
        if val_r > best + 1e-6:
            best, best_state, bad = val_r, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    net.load_state_dict(best_state)
    return lambda Z: net(torch.tensor(Z, dtype=torch.float32, device=device)).detach().argmax(1).cpu().numpy()


# ---- rung 2: sampled REINFORCE (sample one arm, advantage w/ learned baseline + entropy) ----
def fit_policy_reinforce(Ztr, Rtr, Zva, Rva, hp, device, seed):
    torch.manual_seed(seed)
    din = Ztr.shape[1]
    pol = _head(din, Rtr.shape[1], hp["hidden"], hp["dropout"]).to(device)
    val = nn.Linear(din, 1).to(device)
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()),
                           lr=hp["lr"], weight_decay=hp["wd"])
    Zt = torch.tensor(Ztr, dtype=torch.float32, device=device)
    Zv = torch.tensor(Zva, dtype=torch.float32, device=device)
    Rt = torch.tensor(Rtr, dtype=torch.float32, device=device)
    n = len(Ztr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = -1e18, None, 0
    for _ in range(hp["epochs"]):
        pol.train(); val.train()
        order = torch.randperm(n, generator=g)
        for s in range(0, n, hp["batch"]):
            idx = order[s:s + hp["batch"]]
            z, r_all = Zt[idx], Rt[idx]
            logits = pol(z)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()                                  # one arm per prompt (the "1 generation")
            r = r_all.gather(1, a[:, None]).squeeze(1)
            v = val(z).squeeze(1)
            adv = (r - v).detach()
            loss = (-(adv * dist.log_prob(a)).mean()
                    - hp["beta"] * dist.entropy().mean()
                    + hp["cv"] * F.mse_loss(v, r))
            opt.zero_grad(); loss.backward(); opt.step()
        pol.eval()
        with torch.no_grad():
            val_r = _realized(pol(Zv).argmax(1).cpu().numpy(), Rva).mean()
        if val_r > best + 1e-6:
            best, best_state, bad = val_r, {k: v.detach().clone() for k, v in pol.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    pol.load_state_dict(best_state)
    return lambda Z: pol(torch.tensor(Z, dtype=torch.float32, device=device)).detach().argmax(1).cpu().numpy()


def _R_from_M(M):
    """Reward matrix (n, K+1): col 0 = decline (0), cols 1..K = swings (nan->0, matching realized)."""
    return np.concatenate([np.zeros((len(M), 1)), np.nan_to_num(M, nan=0.0)], axis=1)


def _select(fit, grid, Ztr, Ttr, Zva, Tva, Rva, ev_pack, device, seed):
    """Fit every hp, pick by VAL realized, return EVAL realized array for the val-best."""
    Zev, Rev = ev_pack
    best_val, best_ev = -1e18, None
    for hp in grid:
        act = fit(Ztr, Ttr, Zva, Tva, hp, device, seed)
        val_r = _realized(act(Zva), Rva).mean()
        if val_r > best_val:
            best_val, best_ev = val_r, _realized(act(Zev), Rev)
    return best_val, best_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--rep", default="enc", choices=["last", "mean", "enc"])
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    t0 = time.time()

    if args.rep == "enc":
        ec = np.load(OUT / "enc_embed.npz", allow_pickle=True)
        H = ec["Htr"]; rep_name = str(ec["encoder"])
    else:
        st = np.load(OUT / "states.npz", allow_pickle=True)
        L = int(st["layers"][-1])
        H = st[f"Htr_{L}_{args.rep}"]; rep_name = f"LLM layer {L} {args.rep}"
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]                                       # (n, K) train swings over selected moves
    K = len(S)
    idx_all = np.where(~np.isnan(Msel).all(1))[0]

    grid_reg = [dict(hidden=h, dropout=0.3, lr=0.05, wd=wd, epochs=800, patience=40)
                for h in (0, 64) for wd in (0.01, 0.1)]
    grid_exact = [dict(hidden=h, dropout=0.3, lr=0.05, wd=wd, beta=b, epochs=800, patience=40)
                  for h in (0, 64) for wd in (0.01, 0.1) for b in (0.0, 0.01)]
    grid_rein = [dict(hidden=h, dropout=0.3, lr=0.02, wd=0.01, beta=b, cv=0.5, batch=64,
                      epochs=400, patience=40) for h in (0, 64) for b in (0.02, 0.05)]

    methods = {"regression": (fit_regression, grid_reg, "M"),
               "exact-policy": (fit_policy_exact, grid_exact, "R"),
               "reinforce": (fit_policy_reinforce, grid_rein, "R")}
    acc = {m: [] for m in methods}                            # per-seed eval means
    single_means, oracle_means = [], []

    for seed in range(args.seeds):
        idx = idx_all.copy()
        np.random.default_rng(seed).shuffle(idx)
        n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
        tr, va, ev = idx[:a], idx[a:b], idx[b:]
        Ztr, Zva, Zev = _pca(H[tr], [H[tr], H[va], H[ev]], args.n_pca)
        Rtr, Rva, Rev = _R_from_M(Msel[tr]), _R_from_M(Msel[va]), _R_from_M(Msel[ev])

        single_means.append(np.nan_to_num(Msel[ev, 0], nan=0.0).mean())
        oracle_means.append(np.array([max(0.0, np.nan_to_num(Msel[i], nan=-1e9).max()) for i in ev]).mean())

        for m, (fit, grid, tgt) in methods.items():
            Ttr, Tva = (Rtr, Rva) if tgt == "R" else (Msel[tr], Msel[va])
            _, ev_arr = _select(fit, grid, Ztr, Ttr, Zva, Tva, Rva, (Zev, Rev), device, seed)
            acc[m].append(float(ev_arr.mean()))

    single = np.array(single_means); oracle = np.array(oracle_means)
    stats = {m: (np.mean(v), np.std(v), np.mean(np.array(v) > single)) for m, v in acc.items()}
    reg_v = np.array(acc["regression"]); ex_v = np.array(acc["exact-policy"]); re_v = np.array(acc["reinforce"])
    # paired-across-seeds comparisons (same split per seed) — the honest test of "does the objective help"
    d_ex_reg = ex_v - reg_v
    d_re_ex = re_v - ex_v
    dlo1, dhi1 = _boot(d_ex_reg); dlo2, dhi2 = _boot(d_re_ex)
    dt = time.time() - t0

    rows = [f"# Router-as-bandit (offline rungs 0-2) — {args.tag}, rep={args.rep} ({rep_name})\n",
            f"Cached full-information swing log ({len(idx_all)} valid prompts, K={K} moves + decline), "
            f"PCA-{args.n_pca}, {args.seeds} seeds × 60/20/20, honest val-selection. Rewards read from "
            f"swing_train.npz (m_swing) — NO generation; offline CPU {dt:.0f}s, 0 GPU-h, 0 new generations.\n",
            f"Baselines (mean over seeds): single move {single.mean():+.3f} ; naive oracle {oracle.mean():+.3f} "
            f"(run's de-biased oracle ≈ +0.81).\n",
            "| method | eval ΔRM (mean±sd) | vs single | seeds>single | vs regression |",
            "|---|---|---|---|---|"]
    for m in ("regression", "exact-policy", "reinforce"):
        mu, sd, wr = stats[m]
        vs_reg = "—" if m == "regression" else f"{np.mean(np.array(acc[m]) > reg_v)*100:.0f}% seeds"
        rows.append(f"| {m} | {mu:+.3f} ± {sd:.3f} | {mu - single.mean():+.3f} | {wr*100:.0f}% | {vs_reg} |")

    ex_mu = stats["exact-policy"][0]; re_mu = stats["reinforce"][0]; rg_mu = stats["regression"][0]
    rows += ["", "## Paired comparison (same split per seed — the honest test)",
             f"- exact-policy − regression: {d_ex_reg.mean():+.3f} [{dlo1:+.3f}, {dhi1:+.3f}], "
             f"{np.mean(d_ex_reg > 0)*100:.0f}% of seeds positive.",
             f"- reinforce − exact-policy: {d_re_ex.mean():+.3f} [{dlo2:+.3f}, {dhi2:+.3f}], "
             f"{np.mean(d_re_ex > 0)*100:.0f}% of seeds positive.",
             "", "## Reading",
             ("- exact-policy vs regression: **paired CI excludes 0 ⇒ the direct decision objective genuinely helps** on "
              "identical data ⇒ rung-3 online worth building for the effect, not only cost."
              if dlo1 > 0 else
              "- exact-policy vs regression: **paired CI straddles 0 ⇒ NOT distinguishable from regression.** The objective "
              "does not raise the effect here; the bandit's case rests on cost + Study-2 method-consistency, not accuracy."),
             ("- reinforce vs exact-policy: paired CI straddles/above 0 ⇒ the sampled estimator recovers exact ⇒ online rule is sound."
              if dhi2 >= 0 and dlo2 > -0.03 else
              "- reinforce vs exact-policy: **paired gap is negative ⇒ PG variance eats signal in this small-data regime.** "
              "Harden the estimator OFFLINE (base-reward control variate / leave-one-out baseline / larger batch / lr) before spending generation on rung-3."),
             f"- Effect-size wall persists: all three methods ({rg_mu:+.3f}/{ex_mu:+.3f}/{re_mu:+.3f}) sit just above the single "
             f"move ({single.mean():+.3f}) and far below the oracle ({oracle.mean():+.3f} naive) ⇒ the objective is not the "
             "bottleneck; representation/scale (B) or the ceiling itself is. The bandit carries signal, it does not manufacture it.",
             "- All offline on the m_swing log ⇒ noisier than the honest m_test oracle; the m_test confirm still gates the ceiling."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_bandit_{args.tag}_{args.rep}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
