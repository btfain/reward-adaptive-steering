# CLAUDE.md — Durable guardrails for this repo

This is a research build. Read `PLAN.md` for the full staged plan. These rules apply
on **every** turn and override any impulse to move faster.

## Non-negotiable working rules

1. **Stage gating.** Work in the four stages defined in `PLAN.md` (A → B → C → D).
   Each stage has a GREEN criterion. **Do NOT advance past a failing gate.** If a
   gate does not go green, STOP and report what failed and your proposed fix —
   do not proceed to the next stage anyway.

2. **Cheapest-arm-first ordering is deliberate.** Stage A validates the steering
   basis before any policy is built on it. Stage B validates the RM + eval harness
   on the cheapest arm before the RL loop exists. This ordering is what makes a
   failing Stage C debuggable. Do not reorder to "get to the interesting part."

3. **Always log cost.** For every arm, record GPU-hours, peak memory, wall-clock,
   and trainable-param count. Cost numbers must be **measured and logged**, never
   estimated. The cost comparison is a headline result, not an afterthought.

4. **Anti-collapse guard.** Reward gains that come from mode collapse / reward
   hacking are failures, not successes. Always compute a drift/diversity guard
   (KL-to-base or distinct-n / self-BLEU) alongside reward. Cap steering magnitude.
   Keep a KL or diversity regularizer available and use it if collapse appears.

5. **Frozen base model.** The base LM weights are NEVER trained in the steering arms.
   Only the steering policy (and, separately, the LoRA baseline) is trained. If you
   find yourself wanting to touch base weights outside the LoRA baseline, stop.

6. **Method consistency for Study 2.** This code will be reused for a multi-turn
   long-horizon study. Keep the policy class, basis, and algorithm modular and
   environment-agnostic so only the environment + reward swap later. Flag any
   choice that would not carry over.

## Workflow expectations

- **Before writing code for a new stage, state your plan for that stage and wait
  for my go-ahead.** Do not generate files for a stage before I approve its plan.
- Commit to git at each GREEN gate with a clear message (e.g. `stage-A green`).
- Prefer small, inspectable steps over large batches of generated files.
- When something is conceptually hard (basis won't steer, RL variance/instability),
  surface it clearly and concisely — I may take it to a separate reasoning session
  and bring back an answer.
- Keep configs in yaml under `configs/`; don't hardcode hyperparameters in source.

## Defaults are in PLAN.md
Do not silently change the committed defaults (base model, RM, dataset, k, arms).
Swap a default ONLY if a stage GREEN criterion forces it, and say so explicitly
when you do.
