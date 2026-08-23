# Top-k router (FREE, offline) — large_7b: prediction↔selection interpolation

Ridge router (encoder→swing) ranks moves; realized_k = mean max(0, best actual swing in router's top-k). 12 seeds, 75/25 split, K=8. single +0.284; naive(inflated) oracle +1.175; de-biased oracle +0.918. k=1 unbiased; higher k trends toward the inflated naive.

| k (moves generated) | realized ΔRM | Δ from k−1 | % of (naive oracle − single) captured |
|---|---|---|---|
| 1 | +0.628 | — | 39% |
| 2 | +0.866 | +0.239 | 65% |
| 3 | +0.964 | +0.098 | 76% |
| 4 | +1.050 | +0.086 | 86% |
| 5 | +1.105 | +0.055 | 92% |
| 6 | +1.134 | +0.029 | 95% |
| 7 | +1.163 | +0.028 | 99% |
| 8 | +1.175 | +0.013 | 100% |

## Reading
- **top-1 → top-2: +0.239 [+0.201, +0.275]** (captures 65% of the naive headroom vs top-1's 39%).
- **Big top-2 jump ⇒ the router IS a useful narrowing device** — cheap selection at ~2× compute that uses the learned router meaningfully; a real middle-ground result (much better cost story than best-of-n).
- Winner's curse: k=1 is unbiased; the k=K endpoint = naive oracle (inflated above the de-biased ~+0.92). Read the SHAPE / the top-1→top-2 step, not the tail absolute.
