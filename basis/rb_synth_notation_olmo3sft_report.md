# Synthetic notation positive-control — router capability (noise-free, known oracle) — synth_notation_olmo3sft

240 typed prompts, 8 moves, deterministic reward, e5 features, 16 seeds, realized top-1 (route ONE bundle/prompt). Baseline-to-beat = full_rubric (all rules, one prompt).

| arm | realized satisfaction |
|---|---|
| null (no rules) | 0.532 |
| **full_rubric (baseline)** | **0.671** |
| blind (best fixed move = full_rubric) | 0.671 |
| routed — ridge | 0.844 |
| **routed — bandit** | **0.840** |
| oracle (type bundle) | 0.840 |

- routing accuracy (picks correct type bundle): bandit 73%, ridge 64% (chance 12%).
- **routed(bandit) − full_rubric = +0.169 [+0.156, +0.184]** => routing MATERIALLY beats monolithic instruction-stuffing.
- router vs oracle gap = -0.001 (if large despite noise-free known structure => the ROUTER/features/algorithm is the bottleneck).
