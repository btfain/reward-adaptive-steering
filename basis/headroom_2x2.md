# A2 2x2 prototype — conditional steering vs prompting

10 prompts, seeds [0, 1, 2]. Within-modality paired ΔRM vs `none`; a conditional policy may always decline to steer (none = 0). static = best single fixed condition; valid-oracle = per-prompt best picked on seed 0, evaluated on [1, 2] (debiased). RM1 = Skywork-Qwen3-0.6B, RM2 = Skywork-Llama-1B.


## base

| modality | n | static (fixed) | raw-oracle | **valid-oracle RM1** | valid-oracle RM2 |
|---|---|---|---|---|---|
| M1 steering | 24 | +0.34 | +1.92 | **-0.19** | -0.52 |
| M2 prompting | 24 | +0.75 | +2.41 | **+1.03** | +0.64 |
| dense | 8 | +0.88 | +1.48 | **+0.85** | +0.11 |

## large

| modality | n | static (fixed) | raw-oracle | **valid-oracle RM1** | valid-oracle RM2 |
|---|---|---|---|---|---|
| M1 steering | 24 | +0.63 | +2.31 | **+0.98** | +0.63 |
| M2 prompting | 24 | +0.70 | +2.70 | **+1.31** | +1.26 |

## Reading
valid-oracle is the debiased per-prompt headroom a CONDITIONAL policy could capture (0 = conditioning captures nothing beyond declining to steer). Compare within a model: does M1 (steering) reach what M2 (prompting) does? valid-oracle > static means conditioning beats any fixed choice. Agreement between RM1 and RM2 valid-oracle guards against tuning to one RM's noise.
