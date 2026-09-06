# Step 1 — RM noise scale from preference labels (ground truth = dataset labels, NOT a 2nd RM)

RM_A = Skywork/Skywork-Reward-V2-Qwen3-0.6B. 3996 UF-binarized pairs. Label margin m = score_chosen - score_rejected. ΔRM = RM_A(chosen) - RM_A(rejected). RM preference accuracy = the exact BoN-2 selection primitive.

- **Overall preference accuracy A = 0.750** (RM ranks the labeled-better response above the worse one 75.0% of the time).

## Accuracy & ΔRM spread vs label margin
| margin band | n | mean m | accuracy | Var(ΔRM) |
|---|---|---|---|---|
| [0, 0.25) | 478 | 0.00 | 0.506 | 9.747 |
| [0.5, 1) | 753 | 0.50 | 0.608 | 9.637 |
| [1, 2) | 1234 | 1.18 | 0.747 | 11.577 |
| [2, 3) | 581 | 2.15 | 0.857 | 11.232 |
| [3, 5) | 580 | 3.66 | 0.909 | 12.868 |
| [5, 12) | 370 | 5.81 | 0.949 | 16.129 |

## Noise scale sigma_eps (native RM units)
- near-tied band (m<0.5): **sigma_eps ~= 2.208**  (= sqrt(1/2 Var(ΔRM | tied)))
- residual of ΔRM ~ m (binning-free): **sigma_eps ~= 2.396**  (slope 1.114 RM/label-pt)
- fraction of ΔRM variance aligned with the label (R^2) = 0.242 => 76% of RM-difference variance is NOT label-aligned (noise + label-noise + nonlinearity).
- context: pooled RM-score SD across responses = 4.133 (cross-response, not within-prompt).

## Reading
- A=0.750 is the BoN-2 selection reliability against ground-truth labels: ~25% of the time BoN-2 keeps the truly-worse sample. If A on CLEAR pairs (large m) is still well below 1.0, that shortfall is gross RM noise that even a big true gap can't overcome.
- sigma_eps here (RM units) feeds Step 2: real fraction of BoN headroom ~= 1 - sigma_eps^2/sigma_s^2, with sigma_s^2 = within-prompt RM-score variance from a generation pool (next).
- CAVEAT: labels are the dataset's GPT-4 annotations (independent of our RM, but not human) and this sigma_eps is IN-distribution => an OPTIMISTIC (lower) bound on noise for off-distribution BoN winners.

## Secondary cross-check (descriptive, NOT ground truth)
- RM_B (Skywork/Skywork-Reward-V2-Llama-3.2-1B) accuracy vs labels = 0.749 on 3996 pairs; RM_A/RM_B argmax agreement = 0.881.
