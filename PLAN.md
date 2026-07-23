# A-E · Study 1 (Contextual Bandit) — Execution Plan **v2**

> **Revision note (v2).** v1 assumed we could go straight from a validated steering
> basis to a fixed-steering arm that beats SFT. That arm came back **null**: no
> combination of steering vectors clearly beat no-steering against the reward model.
> v2 responds by (a) refusing to treat "add evaluation power" as the sole diagnosis,
> (b) inserting two new stages that must pass before any further real-data search,
> and (c) changing the action space from a dense k-dim vector to a **hierarchical /
> sparse** one. The three-claim logic of the study is unchanged.

Lightweight, interpretable, reward-guided **behavioral control** policy that maps a
prompt's hidden state to a low-dimensional action, holding the base LM frozen.
Single-turn contextual-bandit setting: **no environment, no user simulator, no
trajectory reward.**

Target result: reward competitive with best-of-k and LoRA-RLHF, at **lower training
cost**, with an **interpretable** action.

**Nested experimental logic** (each claim isolated by exactly one arm):
- `fixed-control` vs `SFT` → control in the action space moves reward at all.
- `ours` vs `fixed-control` → *conditioning* the control on the prompt matters.
- `ours` vs `best-of-k` / `LoRA-RLHF` → competitive with standard tools at lower cost.

---

## 0. Where we are, and the three live hypotheses

Stage A succeeded: steering vectors were extracted and validated (each axis produces
a visible behavioral shift). Stage B (fixed-global-steering) returned **mixed/null**
results — nothing clearly better at exploiting the RM than the unsteered SFT
checkpoint.

Do **not** proceed as though this is only a statistical-power problem. Three
hypotheses are live, with very different implications:

| | Hypothesis | Implication if true | Test |
|---|---|---|---|
| **H1** | **Noise.** Effect is real but small vs. RM variance. | Fix measurement: paired scoring, more prompts, variance reduction. | Paired within-prompt scoring (Stage A2) |
| **H2** | **Composition is meaningless.** The weighted sum `Σ aᵢvᵢ` over 6 simultaneous axes is off-manifold / self-interfering; the good region is tiny. | Change the action space to hierarchical/sparse. | Single-axis sweep vs. dense combos (Stage A2) |
| **H3** | **The RM is flat in these directions.** Our axes are *stylistic* (hedge, concise, formal, warm); a preference RM may be substantially invariant to style, caring about helpfulness/correctness/instruction-following instead. | **Existential for Study 1 as specified.** The basis, the RM, or both must change. | Headroom/range diagnostic (Stage A2) |

**H3 is the dangerous one and has not yet been tested.** Test it before any further
search.

---

## 1. Committed defaults (v2)

| Component | Default | Δ from v1 |
|---|---|---|
| Base model | `SmolLM2-1.7B-Instruct` — escalated from 360M on 2026-07-22 (Stage A pole-compliance failures; evidence in `data/compliance/`). Runs on university SLURM cluster. | **corrected** (v2 draft carried the stale 360M default) |
| Reward model | `Skywork-Reward-V2-Qwen3-0.6B` (Apache 2.0). Fallback if signal too weak: `Skywork-Reward-V2-Qwen3-1.7B`. Treated as GIVEN. | **pinned** |
| RM/base compatibility | The RM is scored **independently** of the base model. It does **not** need to share a tokenizer or family — it only needs to ingest `(prompt, completion)` text with correct chat-template formatting. | **relaxed** (v1 over-constrained this) |
| Prompt dataset | UltraFeedback prompts (prompts only) for real-data stages; **synthetic type-conditioned prompts** for Stage B0. | **extended** |
| Control modality | **Two modalities compared:** (M1) activation steering vectors; (M2) natural-language imperatives from a fixed vocabulary. | **new fork** |
| Action space | **Hierarchical / sparse**: choose *which* axis (or none) + *how strongly*. At most 1–2 active axes. NOT a dense 6-dim vector. | **changed** |
| Steering basis k | k = 6 active axes: hedge/assert, elaborate/concise, formal/casual, cautious/direct, warm/neutral, inquire/proceed. challenge/accommodate RETIRED 2026-07-22 after failing the cross-steering prong twice (record in `configs/basis.yaml` retired_axes). | **corrected** (v2 draft carried a stale axis list) |
| best-of-k | k = 8 | — |
| Policy class | Discrete choice over k axes + "none", plus scalar magnitude. Lower-variance and *more* interpretable than dense continuous. | **changed** |
| Steering application | Chosen once per prompt, held FIXED through generation | — |
| RL algorithm | Policy-gradient / GRPO-style against the RM; value baseline | — |

---

## 2. Arms & metrics

| Arm | What | Role |
|---|---|---|
| (i) `SFT` | Frozen base checkpoint, no control | Floor |
| (ii) `best-of-k` | Sample k, keep RM-best | Inference-time baseline |
| (iii-a) `LoRA-RLHF` | LoRA fine-tune, bandit reward | Cost comparison target |
| (iv) `fixed-control` | Single global action (axis + magnitude) tuned vs RM | Isolates conditioning |
| (ours) `conditional` | Learned hidden-state → action policy | The method |

**Metrics (log for every arm):**
- (a) mean held-out RM reward — **always with paired/within-prompt comparison**
- (b) training cost — GPU-hours, peak memory, wall-clock, trainable-param count
- (c) drift / diversity guard — KL-to-base or distinct-n / self-BLEU
- (d) **fluency guard — perplexity under the base model** *(new in v2; see A2)*
- (e) interpretability — which axes the policy uses for which prompt types

---

## 3. Staged execution — gates, in order

### Stage A — Steering basis extraction & validation ✅ COMPLETE
Basis extracted; each axis produces a visible behavioral shift under ±α.

> **v2 caveat:** Stage A validated that steering produces a *visible* shift. It did
> **not** validate that the shift is *fluent*. Carry a perplexity check forward — the
> usable α range is where behavior changes but perplexity stays roughly flat.

---

### Stage A2 — Headroom diagnostic **(NEW — do this before any further search)**

Purpose: measure the **ceiling on exploitable variation** before building any policy
to exploit it, and discriminate H1 / H2 / H3.

For each prompt `x` in a modest set (a few hundred is plenty), over a **single-axis
grid** `{(i, α) : i ∈ [k], α ∈ {−2,−1,+1,+2}}` plus the no-control baseline:

1. Compute `R_best(x)`, `R_none(x)`, `R_worst(x)` over the grid.
2. Report:
   - **Headroom** = `E_x[R_best(x) − R_none(x)]` — oracle gain a perfect conditional policy could capture.
   - **Range** = `E_x[R_best(x) − R_worst(x)]` — total RM sensitivity to the action space at all.
   - **Is `argmax_i` prompt-dependent?** If one axis wins for every prompt, there is no
     conditioning story and fixed-control is the whole method. If it varies by prompt,
     *that variation is the method's opportunity.*
3. Also evaluate a handful of **dense multi-axis combos** and compare their best
   against the single-axis best. If dense ≤ single-axis, that is evidence for **H2**.

**Measurement requirements (these are the point of the stage):**
- **Paired / common-random-numbers scoring.** Score every intervention on the same
  sampled continuation seed where possible; compare *within* prompt, never across.
  Between-prompt RM variance is enormous and will swamp everything. This alone may
  resolve H1 without more data.
- **Perplexity guard.** Log fluency alongside reward. If large α raises perplexity and
  lowers reward, you are measuring *degradation*, not preference.

**Run this in BOTH control modalities on the same prompts:**
- **M1 — steering vectors** (as extracted in Stage A).
- **M2 — natural-language imperatives** (zero-shot, fixed vocabulary: "Be more
  concise," "Hedge more," "Challenge the user's premise," etc.). Needs no extraction,
  so it is nearly free to run.

This comparison directly answers the action-space fork:

| M1 headroom | M2 headroom | Reading |
|---|---|---|
| ~0 | healthy | Basis extraction is the problem, not the premise. Consider M2 as the action space. |
| healthy | ~0 | Steering reaches behavior prompting can't; M1 is the distinctive contribution. |
| ~0 | ~0 | **H3 confirmed** — the RM is flat to behavioral style. Change the basis and/or the RM. Redirects the whole study. |
| healthy | healthy | Earlier null was H1/H2. Proceed with hierarchical action space. |

**GREEN when:** Headroom, Range, and prompt-dependence of `argmax` are measured in
both modalities with paired scoring and a perplexity guard, and H1/H2/H3 is
adjudicated. A null here is a **legitimate, reportable finding** — not a failure to
route around.

---

### Stage B0 — Synthetic positive control **(NEW — run in parallel with A2)**

Purpose: validate the entire B→C machinery on a world where **we know the optimal
conditional policy analytically**, so that a real-data null is interpretable rather
than ambiguous.

**Construction — treat the process as generative:**

1. Define `m` latent types `z ∈ {1..m}`, each associated with a target behavioral
   direction `e_z` over the axes.
2. **Generate short prompts by conditioning on the type**: a language model produces
   each prompt given `z` plus general instruction text. Types are therefore *causally
   upstream* of prompts — no labeling error, and the type is **guaranteed recoverable**
   from the prompt by construction. This is what makes it a true positive control.
3. Define the synthetic reward analytically:

   ```
   R_synth(x, y) = β · ⟨ e_{z(x)} , φ(y) ⟩ + ε
   ```

   where `φ(y)` measures the *realized* behavior of the completion along the axes.
4. **`φ` must be trivially computable and near-error-free** — length for
   concise/elaborate, hedge-word/modal counts for hedge/assert,
   question counts for inquire/proceed. **Do NOT use a learned
   classifier here**; it would reintroduce measurement error into the one place we are
   trying to eliminate it. Learned measurement is for real data only.

**Three dials — the diagnostic grid:**
- **β** (effect size) — how strongly the reward depends on behavior.
- **Degree of type-dependence** — from "one axis is best for everything"
  (no conditioning value) to "every type wants a different axis" (max conditioning value).
- **Type salience** — how overtly the prompt signals its type. High: near-explicit in
  surface form. Medium: overlapping distributions. Low: superficially similar across
  types. Obtained for free by varying how much type-specific instruction text is
  injected at generation.

> **Why salience must be a dial:** if the type is so salient the prompt practically
> announces it, the policy is solving a trivial classification problem and success
> tells us nothing. A positive control that can *only* pass is worthless. The
> interesting result is **where in the 3-D grid the method stops working** — that is
> what tells us which real-world conditions the method requires.

**Implementation notes:**
- **Hold out the generator.** Use a different model (or at least a different
  prompt/temperature regime) to generate synthetic prompts than the base model being
  steered, to avoid familiarity interactions.
- **Keep completions short** (cap max generation length). Reduces `φ` noise and keeps
  the loop iterating in minutes, not hours.

**What to measure:** because the optimum is known analytically, report **regret
against the true optimal conditional policy** — a far stronger claim than "better than
the fixed baseline," and unavailable on real data.

1. Does `fixed-control` recover the population-average best axis? (Sanity.)
2. Does the conditional policy recover the *type-dependent* mapping? (The direct test
   of whether conditioning helps, with a known right answer.)
3. Plot **recovery vs. SNR** and **advantage-over-fixed vs. degree of type-dependence** —
   the latter is the money plot, and the bandit analog of Study 2's inter-turn-dependency knob.

**GREEN when:** the pipeline recovers the known optimal conditional policy at high β /
high type-dependence / high salience, and the recovery curves locate where it fails.
If the pipeline cannot recover a known-recoverable policy, **the bug is in our
machinery, not in the world** — fix it here, cheaply, before touching real data again.

---

### Stage B — Fixed-control arm on real data (revised)
Re-run with the v2 action space (hierarchical/sparse, not dense 6-dim), paired
scoring, and the perplexity guard. Gated on A2 not having confirmed H3.

**GREEN when:** fixed-control beats SFT on paired held-out reward with no diversity
collapse and no perplexity degradation.

### Stage C — Conditional policy + RL loop (revised)
Policy = discrete axis choice (incl. "none") + magnitude. Init at "none" (≈ no-op).
Episodic-bandit RL; value baseline; log reward, KL, perplexity, action entropy, and
action distribution by prompt type.

**GREEN when:** (ours) matches or beats fixed-control on paired held-out reward AND
the learned action varies meaningfully across prompt types.

### Stage D — Baselines & full evaluation table
best-of-k (k=8) and LoRA-RLHF (bandit) to convergence; same metrics + full measured
cost. Master table (arms × {reward, cost, drift, fluency}) + interpretability figure.
≥2 seeds on the headline comparison; optionally a 2nd RM.

**GREEN when:** the master table supports all three nested claims and cost numbers are
logged, not estimated, for every arm.

---

## 4. Repo scaffold (v2 additions marked)

```
ae-study1/
  configs/            # model, RM, dataset, basis, RL hyperparams (yaml)
  data/               # cached prompts, contrastive pairs for basis
    synthetic/        # NEW: type-conditioned prompts + type labels
  basis/              # extracted axis vectors + validation report
  src/
    models.py         # base + RM loading, hooked generation w/ steering
    basis_extract.py  # Stage A
    headroom.py       # NEW: Stage A2 diagnostic (both modalities, paired scoring)
    nl_control.py     # NEW: M2 natural-language imperative vocabulary
    synth_world.py    # NEW: Stage B0 type-conditioned generation + analytic R_synth, φ
    fixed_control.py  # Stage B
    policy.py         # hierarchical/sparse policy (axis choice + magnitude)
    train_rl.py       # Stage C episodic-bandit loop
    baselines.py      # best-of-k, LoRA-RLHF (Stage D)
    evaluate.py       # metrics: reward, cost, drift, fluency, interp
  results/            # tables, figures, logs, seeds
  README.md
```

---

## 5. Forward-consistency note
When Study 2 (long-horizon) reuses this code, the **method stays identical** — same
policy class, same action space, same algorithm — only the environment and reward
swap. The hierarchical/sparse action space adopted in v2 is *more* natural for Study 2
than the dense vector was: it maps directly onto the "small option vocabulary" framing
of the multi-turn MDP. Do not let Study 1 accrue choices that can't carry over.

## 6. Standing epistemic rule
A null result that is **measured properly** is a finding, not a failure. If A2 shows
the RM is flat to behavioral style (H3), say so plainly and redirect — do not tune
until something crosses a threshold. Report the headroom numbers either way.
