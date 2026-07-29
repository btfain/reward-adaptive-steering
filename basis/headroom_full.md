# A2 full run — steering vs prompting, 200 prompts

200 prompts, main seed 0 + winner-validation on [1, 2]. Within-modality paired ΔRM vs none; conditional policy may decline (none = 0). static = best fixed condition (seed 0); valid-oracle = per-prompt winner picked on seed 0 by RM1, evaluated out-of-seed. [95% bootstrap CI over prompts]. RM1 = Qwen3-0.6B, RM2 = Llama-1B.


## base

| modality | n | static (RM1) | **valid-oracle RM1 [CI]** | valid-oracle RM2 [CI] |
|---|---|---|---|---|
| M2 prompting | 24 | +0.27 | **+0.51 [+0.26,+0.79]** | +0.58 [+0.27,+0.91] |

## large

| modality | n | static (RM1) | **valid-oracle RM1 [CI]** | valid-oracle RM2 [CI] |
|---|---|---|---|---|
| M1 steering | 24 | +0.11 | **+0.15 [-0.04,+0.34]** | -0.04 [-0.28,+0.19] |
| M2 prompting | 24 | +0.38 | **+1.08 [+0.85,+1.32]** | +0.86 [+0.59,+1.15] |
| dense | 8 | +0.00 | **-0.11 [-0.25,+0.03]** | -0.27 [-0.46,-0.09] |

## Where headroom lives (7B valid-oracle RM1 by prompt tag)

| tag | M1 steering | M2 prompting | n |
|---|---|---|---|
| long | +0.05 | +0.93 | 58 |
| short | +0.19 | +1.15 | 142 |
| question | +0.18 | +1.02 | 95 |
| statement | +0.13 | +1.14 | 105 |
| open | +0.22 | +1.10 | 149 |
| task | -0.07 | +1.02 | 51 |

## Reading
valid-oracle = debiased ceiling a conditional policy could capture (a learned policy reaches ~55–90% of it, per B0). CI excluding 0 = real headroom. RM1 vs RM2 agreement = not one-RM noise. Compare M1 (steering) vs M2 (prompting) within the 7B; compare 1.7B-M2 vs 7B for the cost story.
