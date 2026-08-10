# Subproject 1 — Reward-Adaptive Steering (method spec)

**One line.** Learn an interpretable, low-dimensional steering *policy* — a prompt-
conditional controller over a jointly-learned low-rank, sparse, orthogonal basis of
residual-stream directions — that optimizes a frozen base model against a given reward
model, at a fraction of RLHF's cost. Setting-agnostic; the single-turn bandit is the
cheapest proving ground.

---

## 1. Novelty boundary (locked 2026-08-10, after full-text read)

"Learn a sparse steering vector" is **occupied**; our four differentiators are not.

| | BiPO (NeurIPS 2024) | YaPO (arXiv preprint, Jan 2026, unrefereed) | **Ours** |
|---|---|---|---|
| Signal | preference pairs, no RM | preference pairs, no RM | **reward model (scalar)** |
| Objective | DPO-style logistic on TF log-probs | same, in a **frozen** SAE latent | **RWR toward the KL-tilt** |
| Structure | single vector / behavior | single sparse vector / behavior | **jointly-learned low-rank set (r>1)** |
| Strength | global, hand-tuned | global, fixed λ | **prompt-conditional controller** |
| SAE | — | frozen (reward never enters dictionary) | reward-in-dictionary = optional (c) |
| Framing | control-to-target-behavior | control-to-target-behavior | **optimize a scalar reward (RLHF alt.)** |

Also cite: **MSRS** (2508.10599, orthogonal per-attribute subspaces, labeled attributes),
**K-Steering** (2505.24535, classifier-gradient, PPLM-like), **ORBIT** (2606.22357,
training-free orthogonal rotation), **SAS / "SAEs decompose steering vectors"**
(2411.08790). None are reward-driven + conditional + jointly-learned-set.

**Headline = the conditional, reward-driven controller** (our Study-1 lineage). Plain
sparse vectors and the SAE-reward variant are interpretability options layered on top.

---

## 2. The learning problem

Frozen base `π_base`. Inject `δ(x)` into the residual stream at layer ℓ; the steered
policy is `π_V(·|x)`. Learn a low-rank basis `V ∈ R^{r×d}` and a controller
`a_θ: h(x) ↦ R^r`, so `δ(x) = a_θ(x)ᵀ V`, to maximize reward under a KL leash:

```
max_{V,θ}  E_x E_{y~π_V(·|x)}[ R(x,y) ]   s.t.  KL(π_V ‖ π_base) ≤ ε.
```

**Hyperparameters (per your framing):** `r` = #directions (deployed capacity), sparsity
per direction, and — decoupled — `n` = training-pool size (supervision richness /
estimator quality; dictates training cost). **`n ≫ r`.**

### Why a bound, and which one
`R` is on sampled `y`; sampling is non-differentiable, so the exact objective has only
the high-variance score-function gradient. We optimize a **differentiable off-policy
surrogate** targeting the KL-optimal tilt `π*(y|x) ∝ π_base(y|x) exp(R/β)` — the
principled ceiling for a KL-constrained lightweight method (best-of-n is merely a
finite-sample proxy for this tilt; `n` sharpens the estimate, it is not the ceiling).

**RWR surrogate (primary).** Per prompt, sample a pool `{y_1..y_n} ~ π_base`, score
`R_k`, fix tilt weights `w_k = softmax_k(R_k/β)` (independent of V → low-variance),
and minimize the weighted teacher-forced NLL of the steered policy:

```
L(V,θ) = − Σ_x Σ_{k=1}^n  w_k(x) · log π_{V,θ}(y_k | x)
         + λ_1 ‖V‖_1  + λ_⊥ ‖VVᵀ − diag‖²_F ,   with ‖δ(x)‖ ≤ ρ‖h‖ (mag cap ≈ KL leash).
```

Fully differentiable, direct gradient, arbitrary batch, full precompute (pool + scores +
base log-probs cached once). Ceiling = the tilt; to exceed base support, refine
on-policy with GRPO (bound-first, break-ceiling-later).

**Variants (config switches, not now):** self-normalized IS reranking (V in the
weights); preference-DPO/BiPO-style logistic (needs pairs, differentiates from BiPO only
via the set + controller); SAE-reward hybrid (reconstruction + L1 + steered-reward on a
reward-active atom subset — distinct from YaPO by *learning* the dictionary under reward;
use the **steered** reward term, never a reward-predictive probe).

---

## 3. Gates

- **S1.1 — Learnability (synthetic positive control).** Analytic reward on cheap φ
  features with a **known** target direction `e*`. Check the RWR learner recovers a
  reward-aligned steering vector (on-policy reward ↑; realized φ moves toward `e*`)
  **before** any real RM. Machinery check, à la B0. *(this is tonight's baseline)*
- **S1.2 — Real RM, single-turn.** Learn against the given RM on UltraFeedback prompts;
  report validated ΔRM (paired, out-of-seed, both RMs) vs the contrastive ~0 baseline
  and vs prompting; add the conditional controller and `r>1`; log measured cost.
- **S1.3 — Cost / interpretability vs RLHF.** vs LoRA-RLHF and best-of-k: reward,
  measured cost, trainable-param count, and **nameability** of the learned directions
  (checked against realized behavior, per B0's "reward-geometry ≠ axis-semantics").

**GREEN (subproject):** a learned conditional steering policy either beats the
contrastive/prompting single-turn baseline at lower cost than RLHF with interpretable
directions, **or** returns a clean, well-measured null bounding single-turn steering's
ceiling. Both are reportable.

---

## 4. S1.1 positive-control design (the baseline experiment)

- **Reward (known):** `R(y) = ⟨e*, φ_std(y)⟩`, `φ = (words, hedge_per100, questions_per100)`
  z-scored over the base pool; default `e* = (−1, +1, 0)` (reward concise **and** hedged —
  a 2-feature target no single contrastive axis matches, so it tests that RWR finds the
  reward-aligned *combination*). No RM, so fast; deterministic; ground-truth optimum.
- **Pool:** SmolLM2-1.7B, layer 16, `n=12` base completions per prompt, short
  (`max_new_tokens=64`, B0 lesson), 64 train / 32 test prompts.
- **Learn:** global `r=1` `V` via RWR + magnitude cap (`ρ=0.15` of measured ref norm,
  the A2 usable band). `l1=0`, `orth=0` at `r=1`.
- **Evaluate:** (i) on-policy ΔR = R(steered gens) − R(base gens) on **test** prompts
  (the honest metric); (ii) recovery — φ shift of steered vs base gens and its alignment
  with `e*`. GREEN-for-S1.1 = ΔR > 0 on held-out prompts and φ moves toward `e*`.

Modules: `src/steer_learn.py` (pool / learn / eval phases), `configs/steer_learn.yaml`,
`scripts/stageS1_synth.sbatch`. Controller `mlp` and `r>1` are coded as stubs, off for
the baseline.
