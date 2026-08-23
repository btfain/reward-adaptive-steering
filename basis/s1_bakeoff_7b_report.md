# Selection bake-off — router-move vs naive best-of-k (REAL, equal compute) — bakeoff_7b

150 held-out prompts, 8 base + 4/move gens, K=8 moves, 768 tok. ΔRM vs mean base. Both = max RM over k real generations (symmetric). Router = ridge (large_7b distilroberta→swing).

| k (gens) | naive best-of-k | router-move best-of-k | move-oracle | random-move |
|---|---|---|---|---|
| 1 | +0.000 | +0.231 | +1.188 | -0.187 |
| 2 | +0.629 | +0.992 | +1.531 | +0.614 |
| 3 | +0.931 | +1.261 | +1.658 | +0.967 |
| 4 | +1.124 | +1.441 | +1.712 | +1.295 |
| 5 | +1.262 | +1.587 | +1.749 | +1.433 |
| 6 | +1.369 | +1.675 | +1.758 | +1.568 |
| 7 | +1.454 | +1.735 | +1.769 | +1.677 |
| 8 | +1.525 | +1.772 | +1.768 | +1.774 |

## Reading
- **router-move best-of-2 +0.992 vs naive best-of-2 +0.629: Δ+0.350 [+0.160, +0.535]**  ⇒ **the basis+router BEATS naive resampling at equal compute** — the cost story holds; a few reward-driven moves > many base samples.
- naive catches router-move-best-of-2 at k=4 (so 2 moves ≈ 4 base samples).
- router vs its bounds: random-move-2 +0.614 (lower) ≤ router +0.992 ≤ oracle +1.531 (upper) — gap to oracle = router ranking headroom.
