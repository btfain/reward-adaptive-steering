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

---

## 5. S1.1 result (2026-08-11): GREEN

RWR learns a reward-increasing steering direction from scratch that generalizes
out-of-sample, CI excluding 0 under two settings (cap 0.15/64tok Δ-R +0.332 [+0.125,+0.591];
cap 0.25/128tok +0.477 [+0.215,+0.780]). Diagnostics: (i) concision is **not** cap/length-
limited — more magnitude amplifies reward via *hedge* while words stay flat, i.e. a single
global direction locks onto the dominant lever and can't serve a second feature; (ii) the
learned direction is **⟂ every contrastive axis** (|cos|≤0.04) while inducing hedging — a
stable reward-geometry≠axis-semantics instance; (iii) 53–66% per-prompt hit rate (global
steering hurts a large minority). All three motivate the conditional low-rank set below.

---

## 6. S1.2 — the conditional controller

**Object.** `δ(x) = a_θ(h(x))ᵀ V`. Read `h(x)` = last-token, *pre-injection* residual state
at layer ℓ (a fixed feature of the frozen base — precompute once, no backprop through the
read). Basis `V ∈ R^{r×d}` sparse + orthogonal, jointly learned with the controller θ by
the same RWR objective; `δ(x)` projected to the magnitude cap per prompt; no forced
normalization on `a`, so the controller can output `a≈0` (**decline to steer** — addresses
the large minority a global vector hurt).

**Interpretability locus.** The method's interpretability is in **`V`** — sparse, orthogonal,
nameable directions plus the "which direction fires for which prompts" map — **not in the
controller**. A linear map on an intermediate residual stream is a function of an
*uninterpretable* input, so linear-vs-MLP is a **capacity / inductive-prior** question (is a
linear readout enough to capture the conditioning, given this RM / dataset / representation
width and depth), not an interpretability one. **Run linear and a shallow MLP in parallel**
(one job, two heads, same basis + objective) and compare as a capacity result.

**Positive control (first, known answer).** The S1.1 world has a prompt-independent target,
so a global vector is already optimal and a controller adds nothing. To validate
*conditioning*, reuse **B0's type-dependent** structure: latent types with *different*
reward-relevant directions per type (signaled via the manifestation dial), so different
prompts want different steering and a global vector provably can't win. Test: does a
conditional rank-r set recover the type-dependent policy (B0's advantage-over-fixed vs
type-dependence money plot, for steering)? Use *movable* φ levers (e.g. type A→hedge+, type
B→question+) so "does conditioning work" isn't confounded with "concision is a hard linear
lever"; keep concise as a secondary probe of whether a dedicated direction cracks it.

**Then the real RM (quick iterate).** Learn `{V, θ}` on UltraFeedback against the given RM
via RWR over base-sample pools; evaluate on-policy validated ΔRM (paired, out-of-seed, both
RMs) vs the contrastive ~0 and prompting baselines; produce the **value-vs-rank curve**
(reward captured by a rank-r conditional policy) and name each learned direction by its
realized behavior. `r` starts at 2 (positive control), then sweeps (n ≫ r maintained).

### 6a. S1.2 synthetic positive control — result (2026-08-11): NULL, testbed diagnosed

Ran the type-dependent positive control at **both** SmolLM2-1.7B and Qwen2.5-7B (steer/read
L16 / L18, rank 2, no-gauge parameterization, penalty-shaped magnitude). Types = sign of the
read state's top-PC projection (recoverable by construction); reward wants A→hedge+, B→hedge−
(opposite poles of a *reachable* lever). Reports: `basis/s1_cond_report.md`,
`basis/s1_cond_7b_report.md`. Cost (measured, A5000): ~29–33 min wall, 9.6–19.8 GB CUDA peak
each.

**Every prior confound eliminated, yet no routing.** Type-separability probe **90% (1.7B) /
94% (7B)** held-out (legible type, no washout); opposite injection directions verified
expressible (no gauge lock); bidirectional reachable lever (no reachability trap). Despite
all that, at **both** scales every conditional arm collapsed to a *type-invariant* output —
routing cos(δ̄A, δ̄B) = +0.82/+1.00 (1.7B) and +1.00/+1.00 (7B); for the mlp/global arms
`a|A` and `a|B` are literally identical. Conditioning value (best conditional − global) =
**+0.015 (1.7B, but at cos +1.00 ⇒ just a stronger global vector, not conditioning)** and
**−0.075 (7B, worse than global)**. This is the pre-registered "both fail with high probe"
branch of the 2×2: **scale is not the missing ingredient.**

**Diagnosis — the testbed, not (necessarily) the method.** Every arm's Δ-R covers zero,
**including global** (1.7B +0.042 [−0.071, +0.157]; 7B +0.049 [−0.067, +0.194]) — unlike
S1.1, where the global learner's CI *excluded* zero. Realized target-lever change is tiny
everywhere (Δhedge ≈ ±0.1 z; the dominant realized change is length, not hedge). So the
synthetic style lever is too weak to move the reward out of noise → **there is no per-type
reward gradient for the controller to route on.** This is S1.1 lesson (c) resurfacing (only
hedge is even weakly steerable at this scale, and only barely). We can defend the narrow
claim — *with a legible type and expressible routing, the controller does not learn to
condition* — but **cannot** separate "controller won't route" from "no gradient to route on,"
because the reachable lever is too weak to create one. That ambiguity is a property of the
synthetic style-feature testbed, not a verdict on the method.

**Decision (not another synthetic patch).** Four fixes have now shown the synthetic style
levers are too weak to be a routing testbed; a fifth (bigger caps / a stronger hand-built
lever) risks manufacturing a control that no longer resembles the real task. Pivot to the
**real RM**, where the reward has genuine variance. The load-bearing question — *does learned
steering beat contrastive ~0?* — is the `steer_rm.py` **global** arm, answerable
*independent of routing*. Re-ask the routing/conditioning question on the real RM only if the
global arm first shows a real gradient exists. `S1.2` stays **not green**; the real-RM run
(`aeS1-rm7b`) is the gate's live continuation, not an advance past it.
