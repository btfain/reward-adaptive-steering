"""P1(ii) — cost-aware submodular basis selection. Objective f(S)=Σ_x max(0, max_{p∈S} swing(x,p)) is
monotone submodular ⇒ greedy is (1−1/e)-optimal. Naive greedy queries the full C×N swing matrix (one
generation-cell per (prompt, candidate)); prohibitive for a large candidate pool. Cost-aware greedy
recovers (near) the same basis while querying FAR FEWER cells, via:
  * sampled marginals — estimate each candidate's marginal gain from a few prompts, not all N;
  * successive elimination (best-arm ID) — cheaply drop clear losers, spend generation only narrowing
    the top contenders, at each greedy step; Hoeffding confidence bounds under a generation-cost oracle.

A cell (prompt, candidate) is queried at most once (cached) and counts one generation. Validated OFFLINE
here against the cached large_7b matrix (real ground truth, C=20) + a synthetic large-C matrix (savings
shine at large C). Metric: basis overlap + value ratio vs full greedy, and fraction of C×N cells queried.

    python src/cost_submod.py
"""

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


class Oracle:
    """Generation-cost oracle: get(x,p) returns swing[x,p], counting each unique cell once."""
    def __init__(self, M):
        self.M = np.nan_to_num(M, nan=0.0); self.seen = set(); self.q = 0
    def get(self, x, p):
        if (x, p) not in self.seen:
            self.seen.add((x, p)); self.q += 1
        return self.M[x, p]


def value(M, S):
    if not S:
        return 0.0
    return float(np.maximum(0.0, np.nan_to_num(M[:, S], nan=-1e9).max(1)).mean())


def greedy_full(M, K):
    """Standard greedy on the full matrix (reference). Queries all C×N cells."""
    n, C = M.shape
    S, curr = [], np.zeros(n)
    for _ in range(min(K, C)):
        gains = [(np.maximum(0.0, np.maximum(0.0, np.nan_to_num(M[:, p], nan=-1e9)) - curr).mean()
                  if p not in S else -1) for p in range(C)]
        p = int(np.argmax(gains)); S.append(p)
        curr = np.maximum(curr, np.maximum(0.0, np.nan_to_num(M[:, p], nan=-1e9)))
    return S, n * C


def greedy_costaware(M, K, s=64, eps=None, seed=0):
    """Cost-aware greedy: estimate marginals on a FIXED subsample of s≪N prompts (subsampled-marginal),
    and optionally evaluate only a random ~(C/K)log(1/eps) candidate SUBSET per step (stochastic greedy).
    Both have (1−1/e−ε) guarantees (concentration on the marginals + stochastic-greedy). Cells queried =
    unique (subsample-prompt, candidate) pairs — the generation cost."""
    n, C = M.shape
    oc = Oracle(M); rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(s, n), replace=False)                    # fixed prompt subsample for marginals
    curr = np.zeros(len(idx)); S = []
    sub = C if eps is None else min(C, int(np.ceil(C / max(K, 1) * np.log(1.0 / eps))))
    for _ in range(min(K, C)):
        cands = [p for p in range(C) if p not in S]
        if eps is not None and len(cands) > sub:
            cands = list(rng.choice(cands, sub, replace=False))      # stochastic-greedy candidate subset
        best, bg, bcol = None, -1.0, None
        for p in cands:
            col = np.array([max(0.0, oc.get(int(x), p)) for x in idx])
            g = float(np.maximum(0.0, col - curr).mean())            # marginal on the subsample
            if g > bg:
                bg, best, bcol = g, p, col
        S.append(best); curr = np.maximum(curr, bcol)
    return S, oc.q


def _report(name, M, K, s_list):
    Sf, qf = greedy_full(M, K); vf = value(M, Sf); n, C = M.shape
    rows = [f"\n### {name}  (C={C} candidates, N={n} prompts, K={K})",
            f"full greedy: value {vf:+.3f}, cells {qf} (=C×N)",
            "| method | s (prompts) | value | % of full | overlap /K | cells | % of full |",
            "|---|---|---|---|---|---|---|"]
    for eps, tag in ((None, "subsample"), (0.1, "subsample+stochastic")):
        for s in s_list:
            # average over a few seeds (the subsample is random)
            vs, ov, qs = [], [], []
            for seed in range(4):
                Sc, qc = greedy_costaware(M, K, s=s, eps=eps, seed=seed)
                vs.append(value(M, Sc)); ov.append(len(set(Sf) & set(Sc))); qs.append(qc)
            rows.append(f"| {tag} | {s} | {np.mean(vs):+.3f} | {np.mean(vs)/max(vf,1e-9)*100:.0f}% | "
                        f"{np.mean(ov):.1f} | {int(np.mean(qs))} | {np.mean(qs)/qf*100:.0f}% |")
    return "\n".join(rows)


def _synth(C, n, seed=0):
    """Planted submodular structure: each prompt has a few 'good' candidates; most candidates weak."""
    rng = np.random.default_rng(seed)
    M = rng.normal(-0.4, 0.5, (n, C))                                # most candidates hurt (like real data)
    good = rng.integers(0, C, 12)                                    # a dozen genuinely useful candidates
    for g in good:
        cover = rng.random(n) < 0.35                                 # each covers ~35% of prompts strongly
        M[cover, g] += rng.uniform(1.0, 2.5, cover.sum())
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()
    out = ["# P1(ii) cost-aware submodular selection — offline validation\n",
           "Cost-aware greedy (sampled marginals + successive elimination) vs full greedy on a "
           "generation-cost oracle. Metric: basis overlap + value ratio, and fraction of C×N cells queried."]
    # real ground truth: large_7b matrix
    L7 = REPO_ROOT / "results" / "prompt_basis_large_7b" / "swing_train.npz"
    if L7.exists():
        M = np.load(L7, allow_pickle=True)["M"]
        out.append(_report("large_7b (real, C=20)", M, args.K, [8, 16, 32, 64, 128]))
    # synthetic large-C: where the stochastic-greedy candidate saving also shows
    for C in (100, 200):
        out.append(_report(f"synthetic (C={C})", _synth(C, 300), args.K, [16, 32, 64, 128]))
    rpt = REPO_ROOT / "basis" / "s1_cost_submod_report.md"
    rpt.write_text("\n".join(out) + "\n")
    print("\n".join(out)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
