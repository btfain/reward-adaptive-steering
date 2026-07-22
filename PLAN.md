# A-E · Study 1 (Contextual Bandit) — Execution Plan

Lightweight, interpretable, reward-guided **steering** policy that maps a prompt's
hidden state to a low-dimensional action in a fixed behavioral basis, holding the
base LM frozen. Single-turn contextual-bandit setting: **no environment, no user
simulator, no trajectory reward.**

Target result: reward competitive with best-of-k and LoRA-RLHF, at **lower training
cost**, with an **interpretable** action.

**Nested experimental logic** (each claim isolated by exactly one arm):
- `fixed-steering` vs `SFT` → steering in the basis moves reward at all (also the earliest pipeline diagnostic).
- `ours` vs `fixed-steering` → *conditioning* the steering on the prompt matters.
- `ours` vs `best-of-k` / `LoRA-RLHF` → competitive with standard tools at lower cost.

---

## 1. Committed defaults (swap only if a stage GREEN criterion forces it)

| Component | Default |
|---|---|
| Base model | `SmolLM2-1.7B-Instruct` — escalated from 360M on 2026-07-22: 360M failed Stage A pole compliance on cautious_direct and inquire_proceed under two instruction strengths (evidence in `data/compliance/`). Runs on university SLURM cluster; 360M-era code unchanged. |
| Reward model | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` (pinned 2026-07-22; fallback if unworkable locally: `OpenAssistant/reward-model-deberta-v3-large-v2`). Scores (prompt, response) text directly, so base-tokenizer compatibility is a non-issue. Treated as GIVEN. |
| Prompt dataset | UltraFeedback prompts (prompts only — completions are generated, stored responses unused) |
| Steering basis k | **k = 6 active axes**: hedge/assert, elaborate/concise, formal/casual, cautious/direct, warm/neutral, inquire/proceed (added 2026-07-22; expected reward profile flips between bandit and long-horizon settings — key Study 2 carry-over). challenge/accommodate RETIRED 2026-07-22 after failing the cross-steering prong twice, incl. with compliance-filtered pairs (see Stage A outcome; full record in `configs/basis.yaml` retired_axes). Collapse rule outcome: cosine-triggered pairs kept on behavioral dissociation evidence — documented deviation in Stage A outcome. |
| Basis extraction | Contrastive activation steering: mean activation difference between contrastive completion pairs, at a chosen mid-to-late residual layer |
| best-of-k | k = 8 |
| Policy class | Markovian MLP head: base-model hidden state (final token, chosen layer) → k-dim action. No recurrence (single turn). |
| Steering application | Chosen once per prompt, held FIXED through generation (within-turn updating deferred) |
| RL algorithm | Simple policy-gradient / GRPO-style update against the RM; value baseline for variance reduction |

---

## 2. Arms & metrics

| Arm | What | Role |
|---|---|---|
| (i) `SFT` | Frozen base checkpoint, no optimization | Floor |
| (ii) `best-of-k` | Sample k, keep RM-best | Inference-time baseline |
| (iii-a) `LoRA-RLHF` | LoRA fine-tune, bandit reward | Cost comparison target |
| (iv) `fixed-steering` | Single global steering vector tuned vs RM | Isolates conditioning; early diagnostic |
| (ours) `conditional` | Learned hidden-state → steering policy | The method |

**Metrics (log for every arm):**
- (a) mean held-out RM reward
- (b) training cost — GPU-hours, peak memory, wall-clock, trainable-param count
- (c) drift / diversity guard — KL-to-base or distinct-n / self-BLEU (so reward gains can't be mode collapse)
- (d) interpretability — which axes the policy uses for which prompt types

---

## 3. Staged execution — four gates, do NOT advance past a failing gate

### Stage A — Steering basis extraction & validation
1. Load base model + RM; confirm generation and RM scoring work on 5 sample prompts.
2. For each of the k=6 axes, assemble contrastive completion pairs (same prompts, two styles). Compute mean activation difference at the chosen layer to get axis vector `v_i`.
3. Sanity-steer: add `+α·v_i` and `−α·v_i` during generation on held-out prompts; verify the intended behavioral shift is visible and RM-measurable.

**GREEN when:** each axis produces a visible, monotone behavioral shift under ±α steering at the chosen scale. Validation is qualitative-first: per-axis side-by-side sample sheets are read (with user sign-off) BEFORE proxy tables are interpreted — proxies quantify monotonicity, they do not discover the effect. If not green, re-pick the layer first, then bump model scale, BEFORE continuing.

**Stage A outcome (signed off 2026-07-22):** GREEN. All 7 axes monotone and qualitatively visible at |α| ≤ 0.2 of residual norm (ref 678, layer 16); ±0.4 degenerates text on every axis → later stages cap |α| ≤ 0.25. Collapse rule: the cosine prong triggered for hedge↔cautious (0.82) and hedge↔warm (0.74) but the cross-steering prong dissociates both (≥2:1 own-vs-other effects) — axes KEPT, an explicit documented deviation favoring the behavioral prong over the geometric one. challenge↔hedge failed cross-steering (+1.4 own vs +1.4 other); the compliance-filtered remedy (83/200 pairs) was tried and also failed (own +1.0 vs hedge +2.7) → challenge RETIRED, **k = 6 final for Study 1**. Premise-challenge appears not linearly extractable at 1.7B by mean-difference; revisit only with a different extraction method or scale. Known residual: steering warm also casualizes (one-directional); warm proxy is warmth-lexicon-only (exclamations belong to the casualness proxy alone). Mild positive steering on cautious/elaborate (+0.1) already beats the unsteered baseline on RM reward.

### Stage B — Fixed-global-steering arm (end to end)
4. Grid/opt search a single global action `a*` over the k-dim basis to maximize mean RM reward on a train split.
5. Evaluate `a*` on held-out prompts; log reward, KL-to-base, diversity.

**GREEN when:** fixed-steering beats SFT on held-out mean reward with no diversity collapse. This validates basis + RM + eval harness on the cheapest possible arm.

### Stage C — Conditional policy + RL loop
6. Implement the MLP policy: hidden state → k-dim action; init near zero (≈ identity / no-steer).
7. Episodic-bandit RL: sample prompt → policy action → steered generation → RM reward → update. Add value baseline; log reward, KL, entropy of actions.
8. Watch for reward-hacking / collapse; cap steering magnitude; keep a KL or diversity regularizer available.

**GREEN when:** (ours) matches or beats fixed-steering on held-out reward AND the learned action varies meaningfully across prompt types (conditioning is doing work).

### Stage D — Baselines & full evaluation table
9. Run best-of-k (k=8) and LoRA-RLHF (bandit) to convergence; log the same metrics + full training cost.
10. Produce the master table (arms × {reward, cost, drift/diversity}) + an interpretability figure (axis usage by prompt cluster).
11. Robustness: repeat headline comparison on ≥ 2 seeds; optionally a 2nd RM to show the method isn't RM-specific plumbing.

**GREEN when:** the master table supports all three nested claims (§ top) and cost numbers are logged, not estimated, for every arm.

---

## 4. Repo scaffold

```
ae-study1/
  configs/            # model, RM, dataset, basis, RL hyperparams (yaml)
  data/               # cached prompts, contrastive pairs for basis
  basis/              # extracted axis vectors + validation report
  src/
    models.py         # base + RM loading, hooked generation w/ steering
    basis_extract.py  # Stage A
    fixed_steer.py    # Stage B
    policy.py         # MLP policy
    train_rl.py       # Stage C episodic-bandit loop
    baselines.py      # best-of-k, LoRA-RLHF (Stage D)
    evaluate.py       # metrics: reward, cost, drift/diversity, interp
  results/            # tables, figures, logs, seeds
  README.md           # how to reproduce each stage
```

---

## 5. Forward-consistency note
When Study 2 (long-horizon) reuses this code, the **method stays identical** — same
policy class, same basis, same algorithm — only the environment and reward swap. Do
not let Study 1 accrue design choices that can't carry over to a multi-turn MDP.
