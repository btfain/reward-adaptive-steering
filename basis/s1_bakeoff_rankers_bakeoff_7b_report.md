# Bake-off — stronger ranker comparison (FREE, same generations) — bakeoff_7b

150 held-out prompts, best-of-k over the SAME cached gens; only the RANKER varies. naive/oracle/random fixed. router = trained on large_7b swings, applied to held-out prompts.

| k | naive | ridge_distil | mlp_distil | ridge_e5 | oracle | random |
|---|---|---|---|---|---|---|
| 1 | +0.000 | +0.241 | +0.422 | +0.315 | +1.180 | -0.285 |
| 2 | +0.629 | +0.980 | +1.056 | +0.983 | +1.535 | +0.629 |
| 3 | +0.931 | +1.262 | +1.315 | +1.278 | +1.656 | +1.029 |
| 4 | +1.124 | +1.446 | +1.482 | +1.473 | +1.711 | +1.268 |
| 5 | +1.262 | +1.585 | +1.609 | +1.580 | +1.747 | +1.443 |
| 6 | +1.369 | +1.673 | +1.677 | +1.671 | +1.766 | +1.576 |
| 7 | +1.454 | +1.729 | +1.746 | +1.734 | +1.777 | +1.682 |
| 8 | +1.525 | +1.767 | +1.775 | +1.772 | +1.771 | +1.776 |

## At k=2 (the headline budget)
- **ridge_distil**: +0.980  (vs naive +0.629, Δ+0.351; captures 39% of random→oracle ranking headroom)
- **mlp_distil**: +1.056  (vs naive +0.629, Δ+0.427; captures 47% of random→oracle ranking headroom)
- **ridge_e5**: +0.983  (vs naive +0.629, Δ+0.354; captures 39% of random→oracle ranking headroom)

## Reading
- best ranker at k=2: **mlp_distil** (+1.056); vs bake-off ridge_distil (+0.980) Δ+0.076.
- a stronger ranker MOVES the k=2 result toward the oracle ⇒ ranker quality is a live lever; worth a better router.
- oracle-2 +1.535 ≈ naive best-of->8 ⇒ the ceiling of router-narrowed selection if the ranker were perfect.
