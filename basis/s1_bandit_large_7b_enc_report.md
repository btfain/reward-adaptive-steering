# Router-as-bandit (offline rungs 0-2) — large_7b, rep=enc (distilroberta-base)

Cached full-information swing log (450 valid prompts, K=8 moves + decline), PCA-40, 12 seeds × 60/20/20, honest val-selection. Rewards read from swing_train.npz (m_swing) — NO generation; offline CPU 13s, 0 GPU-h, 0 new generations.

Baselines (mean over seeds): single move +0.300 ; naive oracle +1.202 (run's de-biased oracle ≈ +0.81).

| method | eval ΔRM (mean±sd) | vs single | seeds>single | vs regression |
|---|---|---|---|---|
| regression | +0.362 ± 0.095 | +0.062 | 75% | — |
| exact-policy | +0.382 ± 0.091 | +0.082 | 67% | 67% seeds |
| reinforce | +0.337 ± 0.087 | +0.037 | 67% | 33% seeds |

## Paired comparison (same split per seed — the honest test)
- exact-policy − regression: +0.020 [-0.013, +0.054], 67% of seeds positive.
- reinforce − exact-policy: -0.045 [-0.078, -0.015], 25% of seeds positive.

## Reading
- exact-policy vs regression: **paired CI straddles 0 ⇒ NOT distinguishable from regression.** The objective does not raise the effect here; the bandit's case rests on cost + Study-2 method-consistency, not accuracy.
- reinforce vs exact-policy: **paired gap is negative ⇒ PG variance eats signal in this small-data regime.** Harden the estimator OFFLINE (base-reward control variate / leave-one-out baseline / larger batch / lr) before spending generation on rung-3.
- Effect-size wall persists: all three methods (+0.362/+0.382/+0.337) sit just above the single move (+0.300) and far below the oracle (+1.202 naive) ⇒ the objective is not the bottleneck; representation/scale (B) or the ceiling itself is. The bandit carries signal, it does not manufacture it.
- All offline on the m_swing log ⇒ noisier than the honest m_test oracle; the m_test confirm still gates the ceiling.
