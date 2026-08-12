# S1.2 diagnostic sweep — steering ceiling: action-space limit vs learning limit

Qwen2.5-7B-Instruct, steer L18, read L18. Reuses the S1.2 real-RM pool (results/steer_rm_7b). ref_norm=397; caps are frac×ref. Held-out ΔRM1 paired vs base, m=4 samples (oracle m=2). RM1=Skywork-Reward-V2-Qwen3-0.6B. **All fixed directions (learned/contrastive/oracle) injected at exactly the cap magnitude** — the favorable case, isolating direction quality from magnitude (so learned here ≥ steer_rm's raw-magnitude global by construction).

Reference (A2, 7B): contrastive ~0 (+0.15); prompting +1.08; best-of-n ceiling **+1.40** (RM1). Base distinct-2 = 0.576. RM2 dropped here (would OOM training); RM1/RM2 agreement already established in the steer_rm run.

| frac | cap | learned train ΔRM1 | learned heldout ΔRM1 [95% CI] | contrastive ΔRM1 | **oracle ΔRM1** [CI] | distinct-2 L/O (base 0.58) | base-NLL steer/base |
|---|---|---|---|---|---|---|---|
| 0.05 | 20 | -0.168 | -0.538 [-1.023, -0.098] | -0.169 | **+1.039** [+0.801, +1.271] | 0.61/0.55 | 0.65/0.54 |
| 0.1 | 40 | -0.469 | -0.838 [-1.159, -0.525] | -1.615 | **+0.175** [-0.195, +0.544] | 0.67/0.61 | 0.96/0.54 |
| 0.15 | 59 | -5.135 | -6.800 [-7.760, -5.935] | -6.003 | **-2.214** [-2.804, -1.653] | 0.45/0.60 | 2.33/0.54 |
| 0.2 | 79 | -2.625 | -3.100 [-3.816, -2.410] | -8.151 | **-5.408** [-6.212, -4.560] | 0.73/0.49 | 2.17/0.54 |

## Capacity side-check — linear conditional at frac 0.1 (cap 40)
| rank | heldout ΔRM1 [95% CI] |
|---|---|
| 2 | -4.695 [-6.070, -3.419] |
| 8 | -1.272 [-1.770, -0.825] |

## Reading key
- **oracle ≈ 0 across the fluent band** ⇒ ACTION SPACE empty: no fluent reward-increasing direction exists in the reward-relevant subspace; learning is not the bottleneck.
- **oracle high but learned-heldout low, train≈heldout low** ⇒ good fluent directions EXIST but are not capturable by a fixed/learned policy ⇒ LEARNING/structure limit.
- **learned-train high but learned-heldout low** ⇒ GENERALIZATION limit specifically.
- **everything only rises where distinct-2 collapses / base-NLL spikes** ⇒ the fluency vise: reward gains are bought with fluency, not fluent steering (anti-collapse guard, rule 4).
- Judge every number against best-of-n above and prompting (+1.08).
