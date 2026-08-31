# Bandit-as-ranker validation (offline, FREE) — large_7b

Does ranking moves by the sample-efficient bandit policy give top-k as good as the offline ridge/MLP ranker (which needs the full matrix)? 12 seeds, 75/25 split, K=8, cached swings as reward oracle. Same top-k eval for all (bias identical) ⇒ fair comparison.

| k | bandit | ridge | mlp | oracle | random |
|---|---|---|---|---|---|
| 1 | +0.357 | +0.024 | +0.339 | +1.141 | -0.225 |
| 2 | +0.713 | +0.553 | +0.708 | +1.141 | +0.403 |
| 3 | +0.873 | +0.786 | +0.869 | +1.141 | +0.671 |
| 4 | +0.968 | +0.903 | +0.966 | +1.141 | +0.823 |
| 5 | +1.036 | +1.005 | +1.035 | +1.141 | +0.920 |
| 6 | +1.069 | +1.064 | +1.088 | +1.141 | +1.007 |
| 7 | +1.114 | +1.107 | +1.124 | +1.141 | +1.076 |
| 8 | +1.141 | +1.141 | +1.141 | +1.141 | +1.141 |

## Reading
- **bandit − MLP (best offline ranker) at k=2: +0.005 [-0.021, +0.028]** (bandit +0.713, mlp +0.708, ridge +0.553).
- bandit − ridge at k=2: +0.160 [+0.115, +0.211] (bandit clearly beats ridge).
- bandit-ranker ≈ the BEST offline ranker (paired CI spans 0) AND beats ridge ⇒ **(iii) = bandit-as-ranker VALIDATED** — top-k-competitive with the full-matrix ranker, but sample-efficient (trains on sampled arms only, no N×K×resamples matrix). Use it.
- All rankers ≥ random and ≤ oracle; gap oracle−best = residual ranking headroom (info-limited).
