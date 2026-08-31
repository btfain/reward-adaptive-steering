# BANNER — reward vs #generations frontier — bakeoff_7b

150 held-out prompts, real best-of-k (max reward over k actual gens). ΔRM vs mean base. Ranker = ridge (≈ validated bandit-as-ranker).

| k (gens) | naive BoN | **router-move** | move-oracle | random-move |
|---|---|---|---|---|
| 1 | +0.000 | +0.236 | +1.179 | -0.285 |
| 2 | +0.629 | +0.982 | +1.532 | +0.625 |
| 3 | +0.931 | +1.258 | +1.655 | +1.023 |
| 4 | +1.124 | +1.442 | +1.710 | +1.270 |
| 5 | +1.262 | +1.584 | +1.745 | +1.442 |
| 6 | +1.369 | +1.669 | +1.759 | +1.574 |
| 7 | +1.454 | +1.733 | +1.770 | +1.682 |
| 8 | +1.525 | +1.770 | +1.767 | +1.774 |

## Rejection sampling (β-swept: expected #gens → reward)
| β | exp. #gens | reward ΔRM |
|---|---|---|
| 2 | 1.93 | +0.563 |
| 1 | 2.93 | +0.926 |
| 0.5 | 4.42 | +1.259 |
| 0.25 | 5.88 | +1.442 |
| 0.1 | 7.04 | +1.511 |

## Headline
- **router-move best-of-2 +0.982 vs naive best-of-2 +0.629: Δ+0.353 [+0.164, +0.546]**  ⇒ router-narrowed selection BEATS naive resampling at equal compute.
- naive matches router-move-2 at k=4 (2 moves ≈ 4 base samples).
- router captures ranking headroom random-2 +0.625 → router-2 +0.982 → oracle-2 +1.532.
- soft-BoN omitted (dominated on reward-vs-#gens; BoN=argmax is the reward-max selection) — it belongs on the reward-vs-KL axis (future extension).
