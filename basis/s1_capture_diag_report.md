# Capture diagnostic — is the routing info-limit representational or fundamental?

Menu = greedy top-8 moves + null (base), budget=best-of-2, 12 seeds, 75/25 split, bandit-as-ranker. captured = (router − random) / (oracle − random) of the best-of-2 headroom.

Random floor +0.762 · Oracle ceiling +1.668 (headroom +0.906).

| features | realized best-of-2 | headroom captured |
|---|---|---|
| generic | +0.910 | 12% |

- _rm_prompt_base pending: needs base gens WITH TEXT (run base_text gen job, then extract_rm_feats --which prompt_base)._
