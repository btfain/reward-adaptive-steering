# Synthetic notation positive-control — router capability (noise-free, known oracle) — synth_notation_v3

480 typed prompts, 14 moves, deterministic reward, e5 features, 16 seeds, realized top-1 (route ONE bundle/prompt). Baseline-to-beat = full_rubric (all rules, one prompt).

| arm | realized satisfaction |
|---|---|
| null (no rules) | 0.374 |
| **full_rubric (baseline)** | **0.839** |
| blind (best fixed move = full_rubric) | 0.839 |
| routed — ridge | 0.884 |
| **routed — bandit** | **0.870** |
| oracle (type bundle) | 0.833 |

- routing accuracy (picks correct type bundle): bandit 48%, ridge 36% (chance 7%).
- **routed(bandit) − full_rubric = +0.031 [+0.028, +0.035]** => routing beats the full-rubric prompt but only MARGINALLY: a strong model self-routes ~24 conditional rules well, so overload barely bites. Scale up constraints / use a weaker instruction-follower / add conflicting rules to widen the gap.
- router vs oracle gap = -0.037 (if large despite noise-free known structure => the ROUTER/features/algorithm is the bottleneck).
