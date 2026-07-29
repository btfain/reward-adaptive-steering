# A2 RM-variance probe — Skywork-V2-Qwen3-0.6B vs -Llama-3.2-1B

Both RMs score the SAME 18486 cached completions. RM1 = Skywork-Reward-V2-Qwen3-0.6B, RM2 = Skywork-Reward-V2-Llama-3.2-1B.


## base

- per-(prompt,seed) ΔRM correlation RM1↔RM2: **+0.83**
- per-condition mean-effect sign agreement: **92%** (all 24), **100%** on 13 movers (|ΔRM1|>0.3)

| condition | ΔRM1 | ΔRM2 | agree |
|---|---|---|---|
| m2:warm_neutral-0.1 | +0.38 | +0.37 | ✓ |
| m2:warm_neutral-0.2 | +0.30 | +0.20 | ✓ |
| m2:cautious_direct-0.1 | +0.27 | +0.25 | ✓ |
| m2:formal_casual+0.1 | +0.25 | +0.44 | ✓ |
| m2:hedge_assert+0.2 | +0.16 | +0.40 | ✓ |
| m2:cautious_direct+0.1 | +0.12 | +0.28 | ✓ |
| m2:elaborate_concise-0.1 | +0.05 | +0.08 | ✓ |
| m2:cautious_direct+0.2 | +0.04 | +0.02 | ✓ |
| m2:inquire_proceed+0.1 | +0.02 | +0.22 | ✓ |
| m2:hedge_assert-0.1 | +0.01 | +0.21 | ✓ |
| m2:warm_neutral+0.1 | -0.03 | +0.00 | ✗ |
| m2:elaborate_concise+0.1 | -0.18 | +0.22 | ✗ |
| m2:hedge_assert-0.2 | -0.28 | -0.36 | ✓ |
| m2:hedge_assert+0.1 | -0.33 | -0.16 | ✓ |
| m2:inquire_proceed-0.2 | -0.37 | -0.32 | ✓ |
| m2:inquire_proceed-0.1 | -0.40 | -0.22 | ✓ |
| m2:cautious_direct-0.2 | -0.43 | -0.42 | ✓ |
| m2:formal_casual-0.1 | -0.57 | -0.67 | ✓ |
| m2:elaborate_concise-0.2 | -0.57 | -0.40 | ✓ |
| m2:formal_casual+0.2 | -0.62 | -0.54 | ✓ |
| m2:warm_neutral+0.2 | -0.64 | -0.74 | ✓ |
| m2:elaborate_concise+0.2 | -0.73 | -0.47 | ✓ |
| m2:formal_casual-0.2 | -2.35 | -2.79 | ✓ |
| m2:inquire_proceed+0.2 | -3.07 | -3.58 | ✓ |

## large

- per-(prompt,seed) ΔRM correlation RM1↔RM2: **+0.93**
- per-condition mean-effect sign agreement: **91%** (all 56), **98%** on 45 movers (|ΔRM1|>0.3)

| condition | ΔRM1 | ΔRM2 | agree |
|---|---|---|---|
| m2:inquire_proceed-0.2 | +0.54 | +0.52 | ✓ |
| m2:cautious_direct-0.2 | +0.42 | -0.04 | ✗ |
| m2:inquire_proceed-0.1 | +0.38 | +0.33 | ✓ |
| m2:hedge_assert-0.1 | +0.35 | +0.09 | ✓ |
| m2:warm_neutral-0.2 | +0.27 | +0.26 | ✓ |
| m2:warm_neutral-0.1 | +0.24 | +0.21 | ✓ |
| m1:warm_neutral-0.05 | +0.15 | +0.05 | ✓ |
| m2:formal_casual+0.1 | +0.11 | +0.15 | ✓ |
| m1:elaborate_concise+0.05 | +0.06 | -0.11 | ✗ |
| m1:cautious_direct+0.05 | +0.01 | -0.04 | ✗ |
| m2:cautious_direct-0.1 | +0.01 | -0.15 | ✗ |
| m2:hedge_assert+0.1 | -0.02 | +0.04 | ✗ |
| m1:inquire_proceed-0.05 | -0.07 | -0.17 | ✓ |
| m1:hedge_assert+0.05 | -0.07 | -0.22 | ✓ |
| m2:elaborate_concise-0.1 | -0.30 | -0.48 | ✓ |
| dense:cem_best | -0.37 | -0.78 | ✓ |
| m1:formal_casual+0.05 | -0.38 | -0.46 | ✓ |
| m1:elaborate_concise-0.05 | -0.38 | -0.29 | ✓ |
| m1:inquire_proceed-0.1 | -0.38 | -0.75 | ✓ |
| m1:inquire_proceed+0.05 | -0.40 | -0.53 | ✓ |
| m1:hedge_assert-0.05 | -0.47 | -0.55 | ✓ |
| m1:cautious_direct-0.05 | -0.49 | -0.50 | ✓ |
| m2:warm_neutral+0.2 | -0.51 | -0.68 | ✓ |
| m1:warm_neutral+0.05 | -0.52 | -0.78 | ✓ |
| m2:inquire_proceed+0.1 | -0.57 | -0.83 | ✓ |
| m2:warm_neutral+0.1 | -0.63 | -1.05 | ✓ |
| m2:hedge_assert+0.2 | -0.65 | -0.66 | ✓ |
| m2:cautious_direct+0.1 | -0.69 | -1.01 | ✓ |
| m2:formal_casual+0.2 | -0.74 | -0.96 | ✓ |
| m2:elaborate_concise+0.1 | -0.74 | -0.63 | ✓ |
| m1:warm_neutral-0.1 | -0.81 | -1.30 | ✓ |
| m1:elaborate_concise+0.1 | -0.86 | -1.38 | ✓ |
| m2:cautious_direct+0.2 | -0.90 | -1.17 | ✓ |
| m2:formal_casual-0.1 | -1.24 | -1.97 | ✓ |
| m1:hedge_assert-0.1 | -1.28 | -1.39 | ✓ |
| m1:cautious_direct-0.1 | -1.34 | -1.69 | ✓ |
| m2:elaborate_concise+0.2 | -1.35 | -1.54 | ✓ |
| m1:elaborate_concise-0.1 | -1.38 | -1.45 | ✓ |
| m1:formal_casual-0.05 | -1.46 | -1.89 | ✓ |
| m2:hedge_assert-0.2 | -1.54 | -1.93 | ✓ |
| m1:formal_casual+0.1 | -1.57 | -1.91 | ✓ |
| m1:cautious_direct+0.1 | -1.61 | -2.19 | ✓ |
| m1:hedge_assert+0.1 | -1.87 | -2.28 | ✓ |
| m2:elaborate_concise-0.2 | -2.02 | -2.13 | ✓ |
| m2:formal_casual-0.2 | -2.25 | -3.06 | ✓ |
| m1:warm_neutral+0.1 | -2.73 | -3.44 | ✓ |
| m1:inquire_proceed+0.1 | -3.75 | -4.65 | ✓ |
| m2:inquire_proceed+0.2 | -4.15 | -5.44 | ✓ |
| dense:prior | -4.33 | -5.31 | ✓ |
| m1:formal_casual-0.1 | -4.58 | -5.40 | ✓ |
| dense:rand4 | -6.83 | -8.85 | ✓ |
| dense:rand0 | -11.13 | -13.65 | ✓ |
| dense:rand2 | -11.56 | -14.81 | ✓ |
| dense:rand5 | -11.80 | -13.98 | ✓ |
| dense:rand3 | -12.30 | -14.57 | ✓ |
| dense:rand1 | -14.28 | -15.70 | ✓ |
