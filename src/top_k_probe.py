"""Top-k router (FREE, offline) — the middle ground between prediction and selection. Top-1 (pure
prediction) is bounded (B1 ≈ single); top-K (try all, select reward-best) = oracle at K× compute. This
asks: if the router proposes its top-k moves and we generate ONLY those k and keep the reward-best, how
fast does realized reward rise with k? A big top-1→top-2 jump ⇒ the router is a useful NARROWING device
(cheap selection: 2× compute, uses the learned router meaningfully) — partially rescues the cost story.
A slow, ~linear rise to k=K ⇒ the router adds nothing as a ranker; you need all n; pure selection.

Offline on cached large_7b: ridge router (encoder→predicted swing vector) ranks moves; realized_k =
mean over held-out prompts of max(0, best ACTUAL swing among the router's top-k). k=1 is unbiased (=our
+0.38); the max-over-k endpoint approaches the NAIVE (winner's-curse-inflated) oracle, so we mark both
the naive and de-biased oracle so the reader sees the inflation envelope. The robust headline is the
top-1→top-2 delta (max of 2 ≈ minimal inflation). Multi-seed. NO generation.

    python src/top_k_probe.py --tag large_7b --seeds 12
"""

import argparse
import json
from pathlib import Path

import numpy as np

from router_bandit import _pca, _boot

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def ridge_predict(Xtr, Ytr, Xte, lam=1.0):
    """Multi-output ridge encoder→swing vector; nan targets imputed to column mean (≈0 swing)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yfill = np.where(np.isnan(Ytr), np.nanmean(np.where(np.isnan(Ytr), np.nan, Ytr), axis=0), Ytr)
    Yfill = np.nan_to_num(Yfill)
    dmu = Yfill.mean(0)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ (Yfill - dmu))
    return Xte @ W + dmu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    ec = np.load(OUT / "enc_embed.npz", allow_pickle=True); H = ec["Htr"]
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]; K = len(S)
    idx_all = np.where(~np.isnan(Msel).all(1))[0]

    curves, singles, naive = [], [], []
    for seed in range(args.seeds):
        idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
        n = len(idx); a = int(0.75 * n)
        tr, ev = idx[:a], idx[a:]
        Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], args.n_pca)
        pred = ridge_predict(Ztr, Msel[tr], Zev)             # (n_ev, K) predicted swings -> ranking
        rank = np.argsort(-pred, axis=1)                     # best-predicted first
        Mev = Msel[ev]
        row = []
        for k in range(1, K + 1):
            topk = rank[:, :k]
            vals = np.array([np.nan_to_num(Mev[i, topk[i]], nan=-1e9).max() for i in range(len(ev))])
            row.append(np.maximum(0.0, vals).mean())          # decline option -> max(0, best-of-topk)
        curves.append(row)
        singles.append(np.nan_to_num(Mev[:, 0], nan=0.0).mean())
        naive.append(np.array([max(0.0, np.nan_to_num(Mev[i], nan=-1e9).max()) for i in range(len(ev))]).mean())
    C = np.array(curves); single = float(np.mean(singles)); naive_or = float(np.mean(naive))
    # de-biased oracle reference (from the run's ceiling set if available)
    deb = None
    ocj = REPO_ROOT / "results" / "bandit_online_b1_1500" / "oracle.json"
    if ocj.exists():
        deb = float(json.load(open(ocj))["oracle_mean"])

    mean_k = C.mean(0)
    # paired deltas between successive k (robust: top-1->top-2 has minimal winner's curse)
    d12 = C[:, 1] - C[:, 0]; d12lo, d12hi = _boot(d12)
    head = naive_or - single
    rows = [f"# Top-k router (FREE, offline) — {args.tag}: prediction↔selection interpolation\n",
            f"Ridge router (encoder→swing) ranks moves; realized_k = mean max(0, best actual swing in router's top-k). "
            f"{args.seeds} seeds, 75/25 split, K={K}. single {single:+.3f}; naive(inflated) oracle {naive_or:+.3f}"
            + (f"; de-biased oracle {deb:+.3f}" if deb else "") + ". k=1 unbiased; higher k trends toward the inflated naive.\n",
            "| k (moves generated) | realized ΔRM | Δ from k−1 | % of (naive oracle − single) captured |",
            "|---|---|---|---|"]
    for k in range(K):
        dfrom = "—" if k == 0 else f"{mean_k[k]-mean_k[k-1]:+.3f}"
        frac = (mean_k[k] - single) / head * 100 if head > 0 else float("nan")
        rows.append(f"| {k+1} | {mean_k[k]:+.3f} | {dfrom} | {frac:.0f}% |")
    rows += ["", "## Reading",
             f"- **top-1 → top-2: {d12.mean():+.3f} [{d12lo:+.3f}, {d12hi:+.3f}]** "
             f"(captures {(mean_k[1]-single)/head*100:.0f}% of the naive headroom vs top-1's {(mean_k[0]-single)/head*100:.0f}%).",
             ("- **Big top-2 jump ⇒ the router IS a useful narrowing device** — cheap selection at ~2× compute that "
              "uses the learned router meaningfully; a real middle-ground result (much better cost story than best-of-n)."
              if d12lo > 0.05 else
              "- **top-2 barely moves ⇒ the router does NOT usefully rank the runner-up** — you need ~all K to capture the "
              "headroom ⇒ pure selection; the router adds little as a search-narrower. Reinforces the single-turn bound."),
             "- Winner's curse: k=1 is unbiased; the k=K endpoint = naive oracle (inflated above the de-biased "
             f"{('~+'+format(deb,'.2f')) if deb else 'oracle'}). Read the SHAPE / the top-1→top-2 step, not the tail absolute."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_top_k_probe_{args.tag}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
