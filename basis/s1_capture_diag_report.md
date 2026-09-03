# Capture diagnostic — is the routing info-limit representational or fundamental?

Menu = greedy top-8 moves + null (base), budget=best-of-2, 12 seeds, 75/25 split, bandit-as-ranker. captured = (router − random) / (oracle − random) of the best-of-2 headroom.

Random floor +0.762 · Oracle ceiling +1.668 (headroom +0.906).

| features | realized best-of-2 | headroom captured |
|---|---|---|
| generic | +0.910 | 12% |
| rm_prompt | +0.821 | 1% |
| rm_prompt_base | +0.855 | 8% |

- RM(prompt) − generic(prompt): **-0.090 [-0.144, -0.029]** ⇒ prompt-only reward features don't beat generic (prompt alone is the limit).
- RM(prompt, base) − generic(prompt): **-0.056 [-0.197, +0.065]** ⇒ even prompt+base doesn't help — the info-limit is fundamental; the small-selected-basis story stands.
