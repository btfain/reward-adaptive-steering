# Top-k router DE-BIASED — b1 ceiling (held-out), router=ridge on large_7b (distilroberta-base)

Per-sample select/score split (6/6 of 12) kills winner's curse. 100 held-out ceiling prompts, 40 split-seeds, K=8. single +0.641; de-biased oracle (all-K) +0.895. realized_k = pick best-of-router's-top-k on SELECT half, score on SCORE half.

| k (generate) | router top-k | random-k (basis) | router−random | % headroom (router) |
|---|---|---|---|---|
| 1 | +0.619 | +0.471 | +0.148 | -9% |
| 2 | +0.807 | +0.639 | +0.169 | 65% |
| 3 | +0.800 | +0.716 | +0.084 | 63% |
| 4 | +0.834 | +0.784 | +0.049 | 76% |
| 5 | +0.876 | +0.807 | +0.069 | 93% |
| 6 | +0.882 | +0.857 | +0.025 | 95% |
| 7 | +0.880 | +0.877 | +0.003 | 94% |
| 8 | +0.895 | +0.895 | +0.001 | 100% |

## Reading
- **top-1 +0.619 → top-2 +0.807: Δ+0.188 [+0.179, +0.197]** (top-1 -9% of headroom, top-2 65%).
- **router top-2 − random-2 = +0.169 [+0.150, +0.186]** (random-2 only -1% of headroom) ⇒ the RANKING, not just the basis, does the work.
- **⇒ the router IS a useful search-narrower** — router-guided best-of-2 gets ~2/3 of the oracle at 2x compute AND beats random-2; the learned router earns its keep as a RANKER even though top-1 is bounded. Real middle ground; partially rescues the cost story.
- Sanity: de-biased top-1 +0.619 ≈ B1 router/exact-policy ceiling; all-K +0.895 ≈ oracle.json +0.895.
