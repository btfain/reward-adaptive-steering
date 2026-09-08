# Synthetic notation positive-control — router capability (noise-free, known oracle) — synth_notation_v3_olmo3sft

112 typed prompts, 14 moves, deterministic reward, e5 features, 16 seeds, realized top-1 (route ONE bundle/prompt). Baseline-to-beat = full_rubric (all rules, one prompt).

| arm | realized satisfaction |
|---|---|
| null (no rules) | 0.459 |
| **full_rubric (baseline)** | **0.692** |
| blind (best fixed move = full_rubric) | 0.692 |
| routed — ridge | 0.726 |
| **routed — bandit** | **0.709** |
| oracle (type bundle) | 0.732 |

- routing accuracy (picks correct type bundle): bandit 68%, ridge 49% (chance 7%).
- **routed(bandit) − full_rubric = +0.017 [+0.007, +0.026]** => routing beats the full-rubric prompt but only MARGINALLY: a strong model self-routes ~24 conditional rules well, so overload barely bites. Scale up constraints / use a weaker instruction-follower / add conflicting rules to widen the gap.
- router vs oracle gap = +0.023 (if large despite noise-free known structure => the ROUTER/features/algorithm is the bottleneck).
