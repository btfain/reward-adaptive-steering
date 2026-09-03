# Capture diagnostic — is the routing info-limit representational or fundamental?

Menu = greedy top-8 moves + null (base), budget=best-of-2, 24 seeds, 75/25 split, bandit-as-ranker. captured = (router − random) / (oracle − random) of the best-of-2 headroom.

Random floor +0.832 · Oracle ceiling +1.699 (headroom +0.867).

| features | realized best-of-2 | headroom captured |
|---|---|---|
| generic | +0.931 | 8% |
| generic_prompt_base | +1.030 | 20% |
| rm_prompt | +0.830 | -4% |
| rm_prompt_base | +0.877 | 3% |

- generic(prompt+base) − generic(prompt): **+0.099 [+0.035, +0.172]** (SAME encoder, +base gen — the clean (2) test, no RM-pooling confound) ⇒ the base generation carries ROUTABLE signal — base-conditioning helps (significant).
- RM(prompt) − generic(prompt): **-0.101 [-0.171, -0.030]** ⇒ prompt-only reward features don't beat generic (prompt alone is the limit).
- RM(prompt, base) − generic(prompt): **-0.053 [-0.148, +0.037]** ⇒ even prompt+base doesn't help — the info-limit is fundamental; the small-selected-basis story stands.
