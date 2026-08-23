# Idea B — route on TRIAL info (prompt + base gen + score) — large_7b (roberta-base, offline)

Does one base generation supply the per-prompt signal the prompt lacked? Frozen roberta, PCA-40, exact-policy, 12 seeds, honest val-selection. single +0.300; naive oracle +1.176 (de-biased ≈ +0.9). Only the INPUT varies.

| routing input | eval ΔRM (mean±sd) | vs single | vs prompt-only (paired) |
|---|---|---|---|
| prompt | +0.383 ± 0.105 | +0.083 | — (baseline) |
| base_gen | +0.371 ± 0.112 | +0.071 | -0.012 [-0.049, +0.021] |
| prompt+gen | +0.378 ± 0.103 | +0.079 | -0.004 [-0.041, +0.030] |
| prompt+gen+score | +0.397 ± 0.114 | +0.097 | +0.014 [-0.030, +0.062] |

## Reading
- **prompt+gen+score vs prompt-only +0.014 [-0.030, +0.062] (CI includes 0)** ⇒ one base gen does NOT add extractable signal ⇒ the trial info the oracle needs is more than a single sample reveals ⇒ strengthens the single-turn bound; selection (Idea A) or multi-turn is the path.
- Context: all arms vs the +1.18 naive oracle — the gap that stays uncaptured is the conditioning that needs richer trial info than one base draw.
