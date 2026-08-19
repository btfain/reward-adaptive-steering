# B0 bandit simulator (offline, free) — large_7b, rep=enc (distilroberta-base)

Cached swing log as an online-bandit simulator (450 prompts, K=8+decline), PCA-40, 12 seeds, hardened REINFORCE (norm-adv + value baseline + batch 128). Offline CPU 4s, 0 GPU-h, 0 generations. Cost translation at MEASURED 2.84s/gen.

Reference (mean over seeds): single +0.300 ; full-info exact-policy +0.370.

## (2) Hardening gate — hardened online REINFORCE vs exact-policy (noiseless, paired)
- hardened final +0.382 ; exact +0.370 ; paired +0.012 [-0.032, +0.054] ⇒ **recovers exact ⇒ online update rule is sound.**

## (1) Sample-efficiency curve — eval ΔRM vs epochs (=generations/prompt), mean over seeds
| epochs (=gens/prompt) | σ=0 | σ=1 |
|---|---|---|
| 1 | +0.073 | +0.061 |
| 2 | +0.159 | +0.128 |
| 3 | +0.210 | +0.189 |
| 5 | +0.253 | +0.219 |
| 8 | +0.279 | +0.239 |
| 12 | +0.303 | +0.281 |
| 20 | +0.328 | +0.306 |
| 30 | +0.371 | +0.313 |
| 40 | +0.382 | +0.348 |

## Measured E (epochs to 90% of the single→exact gain) and implied training cost
- σ=0: **E ≈ 15.4 gens/prompt** (83% of seeds reached target) ⇒ training ≈ 12.1 GPU-h @N=1000, 36.4 GPU-h @N=3000.
- σ=1: **E ≈ 9.0 gens/prompt** (58% of seeds reached target) ⇒ training ≈ 7.1 GPU-h @N=1000, 21.3 GPU-h @N=3000.

## Reading
- **Gate = (hardened ≈ exact) AND (E small at σ=1.0).** If both hold, B1 online is de-risked at the costed budget.
- σ=1.0 is the realistic m=1 online case; σ=0 is the optimistic (averaged-reward) bound. Read E off σ=1.0.
- All rewards are m_swing point estimates ⇒ E is a planning figure; B1 logs the real generations-to-target.
