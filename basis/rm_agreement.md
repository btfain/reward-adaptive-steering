# A2 RM-variance probe — Skywork-V2-Qwen3-0.6B vs -Llama-3.2-1B

Both RMs score the SAME 2460 cached pilot completions. RM1 = Skywork-Reward-V2-Qwen3-0.6B, RM2 = Skywork-Reward-V2-Llama-3.2-1B.


## base

- per-(prompt,seed) ΔRM correlation RM1↔RM2: **+0.88**
- per-condition mean-effect sign agreement: **91%** (all 56), **100%** on 40 movers (|ΔRM1|>0.3)

| condition | ΔRM1 | ΔRM2 | agree |
|---|---|---|---|
| dense:cem_best | +0.88 | +0.42 | ✓ |
| m2:cautious_direct-0.1 | +0.75 | +0.60 | ✓ |
| m2:warm_neutral-0.1 | +0.64 | +0.42 | ✓ |
| m2:formal_casual+0.1 | +0.64 | +0.61 | ✓ |
| m2:hedge_assert+0.2 | +0.51 | +0.56 | ✓ |
| m2:warm_neutral-0.2 | +0.35 | +0.31 | ✓ |
| m1:inquire_proceed-0.1 | +0.34 | +0.12 | ✓ |
| m1:elaborate_concise+0.1 | +0.28 | +0.08 | ✓ |
| m2:cautious_direct+0.1 | +0.28 | +0.38 | ✓ |
| m2:inquire_proceed+0.1 | +0.20 | -0.01 | ✗ |
| m2:hedge_assert-0.1 | +0.19 | -0.14 | ✗ |
| m2:hedge_assert-0.2 | +0.16 | -0.49 | ✗ |
| m2:hedge_assert+0.1 | +0.12 | -0.15 | ✗ |
| m1:cautious_direct+0.1 | +0.03 | -0.46 | ✗ |
| m2:cautious_direct+0.2 | +0.01 | +0.10 | ✓ |
| m2:formal_casual-0.1 | -0.01 | -0.57 | ✓ |
| m2:warm_neutral+0.1 | -0.02 | -0.50 | ✓ |
| m1:warm_neutral-0.1 | -0.12 | -0.25 | ✓ |
| dense:prior | -0.16 | -0.55 | ✓ |
| m1:hedge_assert+0.1 | -0.16 | -1.19 | ✓ |
| m1:formal_casual+0.1 | -0.17 | -0.52 | ✓ |
| m2:elaborate_concise-0.1 | -0.24 | -0.60 | ✓ |
| m2:formal_casual+0.2 | -0.25 | -0.48 | ✓ |
| m2:inquire_proceed-0.1 | -0.38 | -0.65 | ✓ |
| m1:hedge_assert-0.1 | -0.43 | -0.86 | ✓ |
| m2:elaborate_concise-0.2 | -0.47 | -0.84 | ✓ |
| m2:inquire_proceed-0.2 | -0.58 | -0.71 | ✓ |
| m2:elaborate_concise+0.1 | -0.59 | -0.28 | ✓ |
| m2:cautious_direct-0.2 | -0.63 | -0.77 | ✓ |
| m2:elaborate_concise+0.2 | -0.67 | -0.56 | ✓ |
| m1:cautious_direct-0.1 | -0.71 | -0.73 | ✓ |
| m1:warm_neutral+0.1 | -1.00 | -1.77 | ✓ |
| m2:warm_neutral+0.2 | -1.04 | -1.52 | ✓ |
| m1:elaborate_concise-0.1 | -1.15 | -1.78 | ✓ |
| m1:inquire_proceed-0.2 | -1.39 | -2.21 | ✓ |
| m1:inquire_proceed+0.1 | -1.95 | -3.05 | ✓ |
| m1:cautious_direct+0.2 | -2.11 | -3.44 | ✓ |
| m1:elaborate_concise+0.2 | -2.28 | -3.34 | ✓ |
| m2:formal_casual-0.2 | -2.28 | -3.01 | ✓ |
| dense:rand4 | -2.48 | -3.12 | ✓ |
| m1:formal_casual+0.2 | -2.58 | -3.26 | ✓ |
| m1:hedge_assert+0.2 | -2.61 | -3.76 | ✓ |
| m1:hedge_assert-0.2 | -2.72 | -3.72 | ✓ |
| m1:formal_casual-0.1 | -2.76 | -3.36 | ✓ |
| m2:inquire_proceed+0.2 | -3.37 | -4.57 | ✓ |
| m1:elaborate_concise-0.2 | -3.61 | -4.50 | ✓ |
| dense:rand0 | -3.76 | -4.84 | ✓ |
| m1:inquire_proceed+0.2 | -4.17 | -5.73 | ✓ |
| m1:warm_neutral-0.2 | -4.42 | -5.23 | ✓ |
| m1:cautious_direct-0.2 | -4.45 | -6.29 | ✓ |
| m1:warm_neutral+0.2 | -4.51 | -5.75 | ✓ |
| dense:rand2 | -4.63 | -6.10 | ✓ |
| dense:rand5 | -5.04 | -6.98 | ✓ |
| dense:rand3 | -6.58 | -7.59 | ✓ |
| dense:rand1 | -7.44 | -9.39 | ✓ |
| m1:formal_casual-0.2 | -9.12 | -10.56 | ✓ |

## large

- per-(prompt,seed) ΔRM correlation RM1↔RM2: **+0.80**
- per-condition mean-effect sign agreement: **96%** (all 24), **100%** on 20 movers (|ΔRM1|>0.3)

| condition | ΔRM1 | ΔRM2 | agree |
|---|---|---|---|
| m2:hedge_assert+0.1 | +0.70 | +0.83 | ✓ |
| m2:formal_casual+0.1 | +0.59 | +0.64 | ✓ |
| m2:cautious_direct-0.1 | +0.58 | +0.20 | ✓ |
| m2:inquire_proceed-0.1 | +0.56 | +0.60 | ✓ |
| m2:warm_neutral-0.1 | +0.50 | +0.74 | ✓ |
| m2:inquire_proceed-0.2 | +0.44 | +0.66 | ✓ |
| m2:warm_neutral-0.2 | +0.26 | -0.06 | ✗ |
| m2:hedge_assert-0.1 | +0.08 | +0.08 | ✓ |
| m2:cautious_direct-0.2 | -0.16 | -0.67 | ✓ |
| m2:warm_neutral+0.1 | -0.25 | -0.69 | ✓ |
| m2:formal_casual+0.2 | -0.45 | -0.20 | ✓ |
| m2:warm_neutral+0.2 | -0.48 | -0.79 | ✓ |
| m2:hedge_assert+0.2 | -0.61 | -0.52 | ✓ |
| m2:elaborate_concise-0.1 | -0.62 | -0.76 | ✓ |
| m2:cautious_direct+0.1 | -0.63 | -0.86 | ✓ |
| m2:cautious_direct+0.2 | -1.02 | -1.33 | ✓ |
| m2:elaborate_concise+0.1 | -1.16 | -0.82 | ✓ |
| m2:formal_casual-0.1 | -1.54 | -2.26 | ✓ |
| m2:elaborate_concise+0.2 | -1.59 | -1.90 | ✓ |
| m2:hedge_assert-0.2 | -1.63 | -2.09 | ✓ |
| m2:inquire_proceed+0.1 | -1.93 | -2.46 | ✓ |
| m2:formal_casual-0.2 | -2.05 | -2.52 | ✓ |
| m2:elaborate_concise-0.2 | -2.23 | -2.40 | ✓ |
| m2:inquire_proceed+0.2 | -4.05 | -5.33 | ✓ |
