# P1 END-TO-END BANNER — automated pipeline vs best-of-n (held-out) — end_banner_7b

48 HELD-OUT prompts (disjoint from the 96 router-training prompts). Basis = candpool greedy top-8; router = ridge on candpool swings. Real best-of-k, ΔRM vs mean base.

| k (gens) | naive BoN | **router-move (pipeline)** | move-oracle | random-move |
|---|---|---|---|---|
| 1 | +0.000 | +0.099 | +1.108 | -0.003 |
| 2 | +0.613 | +0.876 | +1.522 | +0.741 |
| 3 | +0.919 | +1.251 | +1.662 | +1.118 |
| 4 | +1.120 | +1.450 | +1.749 | +1.342 |
| 5 | +1.270 | +1.564 | +1.779 | +1.508 |
| 6 | +1.389 | +1.693 | +1.814 | +1.633 |
| 7 | +1.488 | +1.768 | +1.820 | +1.735 |
| 8 | +1.575 | +1.836 | +1.816 | +1.835 |

## Rejection sampling (β-swept)
| β | exp #gens | reward |
|---|---|---|
| 2 | 1.97 | +0.521 |
| 1 | 3.10 | +0.904 |
| 0.5 | 4.88 | +1.293 |
| 0.25 | 6.45 | +1.503 |

## Headline
- **pipeline router-move best-of-2 +0.876 vs naive best-of-2 +0.613: Δ+0.246 [-0.067, +0.579]** ⇒ pipeline ≈/below naive — the end-to-end claim is not supported here.
- naive matches pipeline-2 at k=3 (2 router-moves ≈ 3 base samples).
- ranking headroom: random-2 +0.741 → router-2 +0.876 → oracle-2 +1.522.
