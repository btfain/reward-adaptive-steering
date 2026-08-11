# Reward-Adaptive Steering — Execution Plan **v3**

> **Revision note (v3, 2026-08-10).** The Stage A2 full run (200 prompts, out-of-seed
> validation, two RMs) settled the action-space fork, and together with the Rung 1
> reduction (`theory/rung1_reduction.md`) forced a restructure. **Findings:** single-turn
> **steering** has ~zero validated headroom even at 7B (valid-oracle +0.15 [−0.04,+0.34]
> RM1, −0.04 RM2 — CI includes 0); single-turn **prompting** has real headroom at both
> scales (1.7B +0.51, 7B +1.08, CIs exclude 0 under both RMs); **dense composition
> actively hurts** (H2 confirmed); two RM families agree (corr +0.83/+0.93). We were
> **conflating two separable subprojects**; v3 splits them (see **Part II**):
> - **Subproject 1 — Reward-Adaptive Steering** (a *method*): *learn* constrained,
>   interpretable steering vectors for a given base+RM — lightweight/interpretable vs RLHF.
> - **Subproject 2 — The single-turn/multi-turn gap** (a *setting/phenomenon*): demonstrate
>   the greedy-reward gap in a *verifiable* way and close it with *natural prompt directions*;
>   steering-vector learning is out of scope for this track.
> Stages A, A2, B0 are the shared, completed foundation. The old single-turn Stage B→C→D
> pipeline (conditional policy over *contrastive* axes) is **superseded**: its steering arm
> folds into Subproject 1 (now *learned* vectors), its prompting arm and cost baselines feed
> both. The A→B→C→D framing in `CLAUDE.md` is stale — gate-gating now runs *within each track*.

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

**Stage A2 outcome (full run, 2026-07-28→29): the action-space fork is settled.**
Executed as pilot (10 prompts, seeds 0–2) then full run (200 prompts; seed-0 grid +
out-of-seed winner-validation on seeds 1,2; both RMs; bootstrap CIs). Reports:
`basis/headroom_full.md`, `basis/rm_agreement_full.md`. The perplexity guard was
**dropped as a gate** — it conflated intended distribution shift with disfluency
(distinct-2 unchanged while nearly every condition flagged); fluency is not the binding
constraint here.
- **Steering (M1): null.** 7B valid-oracle ΔRM **+0.15 [−0.04,+0.34]** (RM1) /
  **−0.04 [−0.28,+0.19]** (RM2) — CI includes 0 under both. The pilot's +0.98 was
  small-sample selection noise (wide CI); n=200 out-of-seed collapses it. All but two 7B
  steering conditions are reward-negative; the usable |α|≤0.1 band is too weak to help,
  and past it fluency/RM crater. 1.7B steering already failed in the pilot.
- **Prompting (M2): real.** 7B **+1.08 [+0.85,+1.32]** (RM1) / **+0.86** (RM2);
  1.7B **+0.51 [+0.26,+0.79]** (RM1) / **+0.58** (RM2). CIs exclude 0 under both RMs at
  both scales; ~2× the headroom at 7B.
- **Dense (H2): confirmed harmful.** valid-oracle −0.11 (RM1) / −0.27 (RM2); random
  multi-axis combos −11 to −14. Composition goes off-manifold, worse the more axes stacked.
- **RM variance (H1 / RM-noise): ruled out.** Two RM families correlate +0.83 (1.7B) /
  +0.93 (7B) on per-completion ΔRM; 100% / 98% sign agreement on movers.
- **What the RM rewards:** proceed / direct / assertive / complete (the *negative* poles)
  — confident, complete answers. inquire/proceed is single-turn-negative under both RMs
  (the clean hook for Subproject 2's gap).
- **Cost (measured):** 7B 3.75 GPU-h, 1.7B 0.99, RM2 re-score 0.07 (~4.8 total); 7B ≈ 3.8× 1.7B.

**Reading:** H3-for-steering is effectively confirmed single-turn — steering does not move
this RM within the fluent band; prompting is the action space with headroom. This is the
finding that splits the project (Part II).

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

**Stage B0 outcome (signed off 2026-07-24): GREEN.** The pipeline recovers the
recoverable policy up to its feature-set ceiling; both baselines behave correctly;
the dials locate the failure boundaries. Full numbers in `basis/synthB0_report.md`,
`results/synth/synth_results.json`. Key results (β=3, type-dependence d=1 unless noted):
- **Fixed-control sanity: exact.** Captures 1.00 of the type-oracle at d=0 for every β
  (recovers the population-best action) and correctly collapses at d=1 (0.73 vs 4.83).
- **Conditioning works where the world rewards it.** At d=1, β≥1 the conditional policy
  captures 0.55–0.62 of the type-oracle in explicit *and* situational cells vs fixed's
  0.01–0.14, reaching ≈85–92% of the supervised-probe skyline (the feature-set ceiling).
  The money table (conditional − fixed) is monotone in d at every β (β=3: −4.43 / −2.37 / **+0.82**).
- **Failure boundaries as designed.** β=0.3,d=1 drowns everything (SNR floor); the
  none-pool (types randomized → MI=0) *punishes* conditioning (capture −0.37 — acting on
  unrecoverable types is worse than no-op), the built-in negative control; at d=0 the
  conditional pays a ~30% conditioning tax vs fixed.

**Deviations from the plan text above, with rationale:**
- **Salience dial → manifestation dial (explicit / situational / none).** The v1 salience
  dial conflated "preference over the *response*" with "behavior exhibited in the *prompt
  text*": the generator mirrored the preference into prompt style (hedged-type prompts
  themselves hedged, etc.). Replaced by manifestation — explicit (stated), situational (a
  situation in which the behavior is what a good response provides, never named), none (one
  shared neutral pool, types assigned at random → provably unrecoverable). Situational is
  now the real-data analog and the scientifically interesting cell.
- **Generator escalated 1.5B → Qwen2.5-7B-Instruct** (documented default swap): situational
  cells need a stronger instruction follower; templates piloted locally on 1.5B first.
- **k=6 axes (with inquire/proceed), not the plan's original list; 6 types = both poles of
  the three cleanly measurable axes** (length, hedge, question count); the other three axes
  stay in the action space as distractors but never define reward (their proxies are too weak).
- **Metric: value capture, not action-matching regret.** A policy that hedges under type
  uncertainty beats plug-in classification while matching the type-optimal *action* less
  often — correct decision theory, not failure — so recovery is scored by captured value
  ((arm − none)/(oracle_type − none)), against the *type-oracle* (the recoverable ceiling;
  the per-prompt oracle additionally includes generation-sampling luck no policy can predict).

**Carried forward (these bind later stages):**
1. **Overfitting control is mandatory at Stage C scale.** The 2048-dim linear head memorized
   per-prompt generation luck (train +4.7 / test +1.4) until weight decay + early-stopping on
   a stratified val split were added; the algorithm also needed the GRPO-style group-relative
   baseline (PLAN's committed family) — plain REINFORCE + value head collapsed. Both go into Stage C.
2. **The conditioning tax is real.** If the real RM's style preference is weakly prompt-dependent
   (small effective d), the conditional arm can *lose* to fixed. This is what makes A2's
   per-prompt argmax-dependence measurement load-bearing, not decorative.
3. **Reward geometry ≠ axis semantics.** Type-optimal actions were often cross-axis (inquire+0.2
   as the strongest word-count reducer, since the 96-token cap right-censors the elaborate axis;
   and inquire+0.2 partly earns φ via a punctuation artifact). Interpretability claims must be
   checked against realized behavior, not axis names — and this is a preview of Stage C reward hacking.

---

## Part II — Two subprojects (v3)

Stages A, A2, and B0 are the **shared, completed foundation**: the basis (A), the
action-space fork (A2), and the synthetic positive control for the conditional-policy
machinery (B0). What was one single-turn Stage B→C→D pipeline now forks into two
subprojects that had been conflated. Both keep every standing rule in `CLAUDE.md`
(frozen base, measured cost, anti-collapse guard, never advance a failing gate); the
gate discipline now runs **within each track**.

### Subproject 1 — Reward-Adaptive Steering (a method)

**Claim.** We can *learn* interpretable, reward-driven steering vectors for a given
(frozen base, reward model) that are lightweight and interpretable compared to RLHF.
Setting-agnostic; the single-turn bandit is the cheapest proving ground and the
infrastructure already exists (`src/headroom.py`, the steering hooks in `models.py`, RM
scoring, the cost log).

**Shift from the superseded plan.** The action directions are **learned**, not the six
contrastive axes. Learning is constrained for interpretability — **soft low-rank +
sparsity + orthogonality penalties** — and trained against the RM with the B0-carried
recipe (GRPO-style group-relative baseline + weight decay + early stopping; plain
REINFORCE collapsed).

**The honest open question.** A2 showed the *contrastive* single-turn steering headroom is
~0 in the fluent band. Learned directions are not confined to the contrastive span, so
the test is whether a learned constrained direction finds reward-relevant behavior the
hand-picked axes miss. If even a freely-learned constrained vector cannot beat ~0 within
the fluent band, that is a strong, reportable **negative** bounding single-turn steering's
ceiling; if it can, that is the method's win. Either outcome is a result.

**Gates.**
- **S1.1 — Learnability (synthetic positive control).** Reuse the B0 world: can we *learn*
  a constrained steering direction that recovers a known reward-relevant direction?
  Machinery check before the real RM.
- **S1.2 — Real RM, single-turn.** Learn constrained vectors against the given RM on
  UltraFeedback prompts; report validated ΔRM (paired, out-of-seed) vs the contrastive ~0
  baseline and vs prompting; log measured cost.
- **S1.3 — Cost / interpretability vs RLHF.** Against LoRA-RLHF and best-of-k (the old
  Stage D baselines): reward, measured cost, trainable-param count, and **nameability** of
  the learned directions (checked against realized behavior, per B0's "reward-geometry ≠
  axis-semantics").

**GREEN when:** a learned constrained steering policy either (a) beats the
contrastive/prompting single-turn baseline at lower cost than RLHF with interpretable
directions, or (b) returns a clean, well-measured null that bounds single-turn steering's
ceiling.

**S1.1 outcome (2026-08-11): GREEN.** RWR toward the KL-tilt (differentiable teacher-forced
injection, precomputed pool; `src/steer_learn.py`, `basis/s1_synth*_report.md`) learns a
reward-increasing steering direction *from scratch* that generalizes out-of-sample, with
the CI excluding 0 under two settings: baseline cap 0.15/64tok **Δ-R +0.332 [+0.125,+0.591]**,
cap 0.25/128tok **Δ-R +0.477 [+0.215,+0.780]**. Two diagnostics resolved:
- **Concision is not cap/length-limited — it's the single-direction limit.** More magnitude
  amplified reward via *hedge* (+0.22→+0.32) but words stayed flat (≈0.2–0.4%) at both caps.
  A single global direction locks onto the dominant reward lever and can't serve a second
  feature → motivates the conditional low-rank set (S1.2), not a bigger knob.
- **Learned direction ⟂ all contrastive axes** (|cos|≤0.04, incl. hedge_assert −0.01, while
  inducing hedging) — a stable "reward-geometry ≠ axis-semantics" instance (learned ≠ contrastive).
- Hit rate only 53–66% (global steering *hurts* a large minority) → conditioning value, again.

**S1.2 design (spec: `specs/subproject1_spec.md`).** Prompt-conditional controller
`δ(x)=a_θ(h(x))ᵀV` over a jointly-learned rank-r sparse-orthogonal basis. **Interpretability
lives in `V` (nameable directions + which-fires-for-which map), NOT in the controller** — a
linear map on an intermediate residual stream is a function of an uninterpretable input, so
linear-vs-MLP is a *capacity* question, run **both in parallel**. Positive control uses B0's
**type-dependent** targets (a global vector provably can't win) before the real RM.

### Subproject 2 — The single-turn / multi-turn gap (a setting)

**Claim.** Single-turn reward optimization is myopic; the gap is real and material; and
simple interpretable planning over **natural prompt directions** closes most of it.

**Theoretical spine.** `theory/rung1_reduction.md` (Rung 1) — greedy maximization of a
per-step reward is not return-optimal (fully observed; value-of-information the salient
instance); A2 instantiates a strictly-positive-regret regime for a real base+RM pair;
longer context cannot close it.

**Deliberate scope.** Action space = **natural prompt directions only** (the planning-move
axes, articulated as prompts). Steering-vector learning is **out of scope here** — the gap
result must not hinge on the harder method. Reward = a **deterministic verifiable rubric
check**, never a hand-crafted "good conversation" proxy.

**Gates.**
- **Rung 1 — Reduction. ✅ DONE** as a formal note (`theory/rung1_reduction.md`).
- **Rung 2 setup — planning-move axes.** Converge on the substantive, beyond-style
  planning-move set, written as prompts. *(next task)*
- **Rung 2a — Verifiable gap + simple planning (oracle basis).** A clarification-gated
  synthetic environment with a hidden, **checkable** rubric composed of the planning-move
  types (ports B0's manifestation dial as the revelation structure). Demonstrate (i) the
  greedy/deployed policy leaves ground-truth verifiable reward on the table — the gap, in
  units of task success (the "and it matters" experiment Rung 1 points to), and (ii) a
  simple conditional prompt-adapter planner closes most of it.
- **Rung 2b — Learn the action space from reward (deferred).** Discover the planning-move
  directions from the multi-turn reward rather than assuming them. This is where the two
  subprojects **reconnect**: Subproject 1's constrained-learning method, applied where
  reward finally has multi-turn variance.
- **Rung 3 — Real-world validation.** Validate the demonstrated gap and its closure on real
  human–AI interaction logs (external validity; the synthetic env is the positive control).
  Method TBD.

**GREEN (Rung 2a):** the gap is exhibited in ground-truth verifiable reward, a simple
prompt-adapter planner closes a substantial fraction of it, and the greedy baseline is
provably myopic per Rung 1.

> **On the old "Study 2."** The long-horizon study deferred in v2 *is* Subproject 2, now
> promoted to a first-class track with its own theory and gates rather than a code-reuse
> afterthought.

---

## 4. Repo scaffold (v2 additions marked)

```
ae-study1/
  theory/             # NEW (v3): rung1_reduction.md — reduction spine of Subproject 2
  configs/            # model, RM, dataset, basis, RL hyperparams (yaml)
  data/               # cached prompts, contrastive pairs for basis
    synthetic/        # type-conditioned prompts + type labels (B0)
  basis/              # extracted axis vectors + validation + headroom reports
  src/
    models.py         # base + RM loading, hooked generation w/ steering
    basis_extract.py  # Stage A
    headroom.py       # Stage A2 diagnostic (both modalities, paired scoring)
    nl_control.py     # M2 natural-language imperative vocabulary
    synth_world.py    # Stage B0 type-conditioned generation + analytic R_synth, φ
    synth_learn.py    # Stage B0 offline learning (GRPO + early stop)
    rm_agreement.py   # A2 second-RM variance probe
    steer_learn.py    # NEW (v3, Subproject 1): learn constrained steering vectors
    multiturn_env.py  # NEW (v3, Subproject 2 Rung 2a): verifiable clarification-gated env
    baselines.py      # best-of-k, LoRA-RLHF (Subproject 1 S1.3 cost comparison)
    evaluate.py       # metrics: reward, cost, drift, fluency, interp
  results/            # tables, figures, logs, seeds
  README.md
```
> `fixed_control.py` / `policy.py` / `train_rl.py` from the v2 scaffold are **superseded**
> (conditional policy over *contrastive* axes); their reusable pieces move into
> `steer_learn.py`.

---

## 5. Forward-consistency note
Subproject 2 *is* the long-horizon study, now first-class rather than a code-reuse
afterthought. The two subprojects reconnect at **Rung 2b**: the constrained
vector-learning method of Subproject 1, applied where multi-turn reward gives it
variance. Keep Subproject 1's learner environment-agnostic (policy class, basis,
algorithm modular) so it drops into the multi-turn env unchanged. The natural
prompt-direction action space of Subproject 2 maps directly onto the "small option
vocabulary" framing of the multi-turn MDP. Do not accrue choices in either track that
can't carry over.

## 6. Standing epistemic rule
A null result that is **measured properly** is a finding, not a failure. If A2 shows
the RM is flat to behavioral style (H3), say so plainly and redirect — do not tune
until something crosses a threshold. Report the headroom numbers either way.
