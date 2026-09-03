# P1 menu-size sweep — does small-basis selection earn its keep? (offline, FREE)

Candpool swing matrix 96×220, cached-swing reward oracle, 8 seeds, 75/25 split, bandit-as-ranker (epochs=60), PCA-40 distilroberta-base features. Realized best-of-2 vs random-routing floor vs oracle ceiling, as the menu grows. **The null move (base, swing 0) is an explicit menu arm competing for a generation slot.**

## Greedy menu (submodular top-K, selected on the train split)
_oracle = per-prompt menu max (with null: floored at base) = the best-of-k ceiling for ANY k. bandit@2(no-null) = the old arm without a base fallback, for contrast._
| K | bandit@2 | bandit@2 (no-null) | random@2 | oracle ceiling |
|---|---|---|---|---|
| 2 | +0.974 | +0.986 | +0.861 | +1.186 |
| 4 | +0.922 | +0.958 | +0.874 | +1.482 |
| 8 | +0.917 | +0.903 | +0.743 | +1.683 |
| 16 | +0.803 | +0.806 | +0.603 | +1.950 |
| 32 | +0.842 | +0.766 | +0.708 | +2.178 |
| 64 | +0.795 | +0.812 | +0.442 | +2.304 |
| 128 | +0.807 | +0.782 | +0.547 | +2.488 |
| 220 | +0.707 | +0.618 | +0.425 | +2.595 |

## Random menu (a K-subset of the pool, no selection)
| K | bandit@2 | random@2 | oracle ceiling |
|---|---|---|---|
| 2 | +0.573 | +0.406 | +0.760 |
| 4 | +0.582 | +0.504 | +1.114 |
| 8 | +0.653 | +0.475 | +1.303 |
| 16 | +0.619 | +0.457 | +1.655 |
| 32 | +0.725 | +0.546 | +1.891 |
| 64 | +0.733 | +0.325 | +2.154 |
| 128 | +0.777 | +0.398 | +2.387 |
| 220 | +0.658 | +0.522 | +2.595 |

## Realized bandit best-of-k vs menu size (greedy menu)
| K | bandit@1 | bandit@2 | bandit@4 |
|---|---|---|---|
| 2 | +0.411 | +0.974 | +1.186 |
| 4 | +0.370 | +0.922 | +1.353 |
| 8 | +0.398 | +0.917 | +1.343 |
| 16 | +0.300 | +0.803 | +1.188 |
| 32 | +0.261 | +0.842 | +1.218 |
| 64 | +0.226 | +0.795 | +1.236 |
| 128 | +0.232 | +0.807 | +1.164 |
| 220 | +0.234 | +0.707 | +1.121 |

## Keep-everything + train harder (full menu K=220, greedy=random=all cols)
| epochs | bandit@2 |
|---|---|
| 60 | +0.587 |
| 200 | +0.647 |
| 400 | +0.740 |

## Reading
- realized bandit@2 is maximized at **K=2** (greedy menu).
- best small menu (K=2) − full menu (K=220): **+0.267 [+0.194, +0.332]** ⇒ selecting a small basis beats keeping the pool at equal deploy compute — **stage (ii) earns its keep**.
- oracle ceiling vs bandit@2 at K=220: +2.595 vs +0.707 — the routing gap on the full menu (info-limit + exploration cost); if bandit@2 collapses toward random@2 as K grows, the big menu is unrouteable.
- train-harder at K=220: +0.587 (60ep) → +0.740 (400ep) ⇒ more training DOES help the full menu.

_Mechanism note: a larger menu raises oracle@k (more headroom) but the bandit must explore more arms with the same data; the bandit@2−random@2 gap vs K shows whether routing keeps up._
