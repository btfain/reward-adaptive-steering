# Synthetic notation positive-control — router capability (noise-free, known oracle) — synth_notation_v1

240 typed prompts, 8 moves, deterministic reward, e5 features, 16 seeds, realized top-1 (route ONE bundle/prompt). Baseline-to-beat = full_rubric (all rules, one prompt).

| arm | realized satisfaction |
|---|---|
| null (no rules) | 0.505 |
| **full_rubric (baseline)** | **0.848** |
| blind (best fixed move = full_rubric) | 0.848 |
| routed — ridge | 0.899 |
| **routed — bandit** | **0.897** |
| oracle (type bundle) | 0.894 |

- routing accuracy (picks correct type bundle): bandit 73%, ridge 33% (chance 12%).
- **routed(bandit) − full_rubric = +0.049 [+0.043, +0.056]** => routing beats the full-rubric prompt but only MARGINALLY: a strong model self-routes ~24 conditional rules well, so overload barely bites. Scale up constraints / use a weaker instruction-follower / add conflicting rules to widen the gap.
- router vs oracle gap = -0.003 (if large despite noise-free known structure => the ROUTER/features/algorithm is the bottleneck).
