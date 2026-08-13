# S1.2 — per-prompt free-δ reachability probe (rank relaxed)

Qwen2.5-7B-Instruct, steer L18, read L18. Per prompt: unconstrained full-rank δ∈R^d learned by RWR on its own pool (results/prompt_basis_detrunc_7b), 40 steps, evaluated on-policy at exactly the cap. ref_norm=397. RM1=Skywork-Reward-V2-Qwen3-0.6B, m=4 samples.

Denominator: best-of-n ceiling **+1.37** (RM1); prompting ref +1.08; A2 contrastive ~0. Base distinct-2 = 0.653.

| frac | cap | on-policy ΔRM1 [95% CI] | frac of best-of-n | surrogate ΔL (mean) | distinct-2 |
|---|---|---|---|---|---|
| 0.05 | 20 | +0.406 [+0.272, +0.535] | +0.30 | +99.76 | 0.58 |
| 0.1 | 40 | +0.162 [-0.019, +0.342] | +0.12 | +108.47 | 0.52 |

Fluency guard at frac 0.1: base-NLL steered 0.59 vs base 0.50 (spike ⇒ gains bought with fluency).

## Factorization of the per-prompt optima (conditional on headroom existing)
- Effective rank of the 170 per-prompt δ's: **68** dirs for 50% variance, **144** for 90% (of d-dim).
- Predictability from h(x) (ridge, held-out): **R²=-0.24**, mean cosine **+0.03**.
- Low rank + high R²/cos ⇒ a basis+controller could reconstruct these ⇒ S1.2's joint failure was OPTIMIZATION (staged solve-then-factor is viable). High rank / low R² ⇒ the low-rank+controller design was mis-specified.

## Reading key
- **on-policy ΔRM ≈ 0 while fluent** ⇒ reachability WALL: even per-prompt full-rank fluent steering can't move the RM ⇒ steering fundamentally bounded here ⇒ pivot with a clean null.
- **ΔRM a large fraction of best-of-n** ⇒ reachable reward-increasing dirs EXIST ⇒ our global/low-rank/controller failures were LEARNING/design, not reachability ⇒ worth improving.
- **surrogate ΔL large but ΔRM ≈ 0** ⇒ TF-surrogate optimized but doesn't transfer on-policy ⇒ objective/on-policy GAP (motivates on-policy RL), not a reachability wall.
- Judge against best-of-n and prompting (+1.08); watch the fluency guard for collapse.
