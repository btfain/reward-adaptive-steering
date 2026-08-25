# Stage 1 completion plan — from validated mechanism to a finished pipeline + theory

*(Living plan; reconcile gates with PLAN.md's S1.x. Companion to PROBLEM.md, which is the formal spine.
Discipline: cheapest-first, each phase has a GREEN criterion, cost measured+logged, no advancing past a
failing gate.)*

## What is already established (do not redo)
- **Mechanism:** prediction-from-prompt is BOUNDED (B1, paired); **selection** works — router-narrowed
  best-of-2 beats naive best-of-n at equal compute (bake-off, robust across rankers). Steering
  comprehensively out. Formal objective + win-condition in PROBLEM.md.
- **Infra reusable:** swing computation, greedy submodular (plain), bandit/offline routers, top-k bake-off
  harness, offline simulator, de-biasing, multi-seed, cost logging.

## The pipeline (three stages), each EVALUATED individually + an end-to-end banner

**Banner result (the headline figure):** full pipeline end-to-end — auto-discovered candidates →
cost-aware basis → trained ranker → top-k at test time — on the **reward-vs-#generations frontier vs
naive best-of-n** (and vs best-of-n variants: soft-BoN, rejection sampling). Secondary axis: reward-vs-KL.

### (i) Candidate discovery — *currently abstracted (hand-curated file); must become real*
- **Build:** LLM-from-signal generator — feed an LLM contrastive signal (high- vs low-reward completions,
  or preference pairs) and have it propose procedural "moves"; verify (does the move parse / apply) + dedup
  (semantic). Produces a candidate pool of size C.
- **Evaluate (i):** downstream basis value of auto-generated pool vs hand-curated (holding ii,iii fixed);
  pool diversity/coverage; sensitivity to C. GREEN: auto pool ≥ curated pool in basis value at matched C.

### (ii) Cost-aware submodular basis selection — *currently plain greedy on a fully-evaluated matrix; the
algorithmic contribution is unbuilt*
- **Build:** lazy-greedy (CELF) + **sample-based marginal estimation under a generation-cost oracle** —
  estimate each candidate's marginal gain from few generations with concentration bounds (best-arm-ID
  flavor), refine only contenders. Screening-by-average is valid only for move #1 (marginals are
  context-dependent past that).
- **Evaluate (ii):** (a) *quality* — value-vs-K vs exhaustive/oracle selection; (b) *cost* — generations
  used vs naive "evaluate-everything" selection (the efficiency headline); ablate C and K. GREEN:
  cost-aware reaches ≥(1−1/e−ε) of oracle basis value at a large measured generation saving.

### (iii) Ranking model + top-k — *have a working router; finalize + ablate*
- **Decision:** for *selection* the offline ranker (train on swing/preference matrix → rank) is simpler and
  sufficient; keep the online bandit as the method-consistency bridge to Study 2, but the banner uses the
  offline ranker. Ablate ranker architecture/features (ridge / MLP / encoder) and **top-k**.
- **Evaluate (iii):** fraction of random→oracle ranking headroom captured; reward-vs-k frontier vs BoN;
  the win-condition per prompt (mode vs tail). GREEN: router top-2 > naive best-of-2 (have: +0.35), and a
  characterized k where routed selection Pareto-dominates BoN.

## Reward-model vs reward-free (preference-only) — run the WHOLE pipeline both ways
- **RM mode:** swing = RM(move) − RM(base); ranker on RM targets. (current)
- **Reward-free mode:** only preference pairs. (i) generator reads preference pairs; (ii) move "value" =
  win-rate of move-output vs base-output under pairwise comparison (a preference/BT model or held-out
  judge); (iii) ranker trained with pairwise/ranking loss. GREEN: reward-free pipeline matches RM pipeline
  on the banner frontier (within CI) — establishes RM-free applicability (DPO-adjacent).

## Robustness / generality matrix (prioritized — NOT the full cross-product)
Axes: **generative model** × **prompt distribution** × **reward/preference source**. Core = current
(Qwen2.5-7B × UltraFeedback × Skywork). Prioritized additions (a diagonal + key cells, not all cells):
- +1–2 base models (e.g. Llama-3.1-8B, Mistral-7B) — pipeline robust across generators?
- +1–2 prompt distributions (e.g. a chat/HH set, a domain set) — robust across input distributions?
- +1 alternate RM family AND +≥1 **specialized preference dataset** (population/purpose-specific) — needed
  for the dependency question below.
GREEN: banner (routed selection > BoN) holds in every run attempted.

## Dependency question — generic vs specific (secondary, but potentially the most interesting)
- **Basis-transfer matrix:** discover basis on (RM_A, model_A), evaluate on (RM_B, model_B). Measure basis
  overlap + value retained under transfer.
- **Hypothesis:** generic UltraFeedback → similar, transferable bases ("be thorough / reason step by step")
  ⇒ likely null; **specialized** preference data (a specific population/purpose) → divergent, informative
  bases that do NOT transfer ⇒ the interesting positive. Include ≥1 specialized set to test this.

## Theory (developed in parallel; GPU-independent)
- **T1 (stage ii):** cost-aware greedy retains (1−1/e−ε) with sampled marginals — concentration/sample-
  complexity bound under a generation-cost oracle (best-arm-ID). *Primary theorem.*
- **T2 (stage iii):** bound realized best-of-k reward as a function of ranker top-k recall (P[true-best ∈
  top-k]); ties ranking quality to the banner metric.
- **T3 (overall):** under a mixture-of-modes model of the reward-tilted target π*, characterize when routed
  selection Pareto-dominates best-of-n on reward-vs-cost (formalizes the mode-vs-tail win-condition).
- **T4 (reward-free):** consistency of preference-based move value with RM-based value under a BT model.

## Sequencing (cheapest-first, gated) — proposed
1. **P1 — finalize + evaluate the pipeline on the core setting** (reuses existing data/harness):
   (iii) ablations first (cheapest, mostly offline) → (ii) cost-aware algorithm (offline on cached swings +
   cost oracle; where T1 lives) → (i) candidate generator (generation + rerun select/rank).
   *Deliver: per-step evals + end-to-end banner on the core setting.*
2. **P2 — reward-free variant** on the core setting. *Deliver: RM-free matches RM.*
3. **P3 — robustness** across the prioritized model/data/RM cells (+ specialized preference set).
4. **P4 — dependency/transfer matrix.**
5. **Theory T1–T4** in parallel throughout (T1 with P1-ii, T2 with P1-iii, T3 with the banner, T4 with P2).

## What I need decided before implementing P1
- **Scope/budget:** how many robustness cells (P3) — the full cross-product is a lot of GPU; propose a
  diagonal + 2–3 key cells.
- **Model/data/RM choices:** which 1–2 extra base models, prompt distributions, RMs, and which specialized
  preference dataset (this one drives the dependency story).
- **Theory weight:** co-equal thrust (aim for T1 as a real theorem) vs supporting (assumptions-heavy
  sketches alongside experiments)?
- **Banner baselines:** naive BoN only, or also soft-BoN + rejection sampling (cheap to add)?
