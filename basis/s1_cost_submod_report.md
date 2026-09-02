# P1(ii) cost-aware submodular selection — offline validation

Cost-aware greedy (sampled marginals + successive elimination) vs full greedy on a generation-cost oracle. Metric: basis overlap + value ratio, and fraction of C×N cells queried.

### large_7b (real, C=20)  (C=20 candidates, N=450 prompts, K=12)
full greedy: value +1.253, cells 9000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 8 | +1.159 | 92% | 7.0 | 160 | 2% |
| subsample | 16 | +1.181 | 94% | 7.8 | 320 | 4% |
| subsample | 32 | +1.216 | 97% | 8.8 | 640 | 7% |
| subsample | 64 | +1.223 | 98% | 8.8 | 1280 | 14% |
| subsample | 128 | +1.235 | 99% | 9.0 | 2560 | 28% |
| subsample+stochastic | 8 | +1.186 | 95% | 7.5 | 156 | 2% |
| subsample+stochastic | 16 | +1.194 | 95% | 8.2 | 316 | 4% |
| subsample+stochastic | 32 | +1.226 | 98% | 9.2 | 624 | 7% |
| subsample+stochastic | 64 | +1.229 | 98% | 9.0 | 1280 | 14% |
| subsample+stochastic | 128 | +1.217 | 97% | 8.5 | 2528 | 28% |

### candpool_7b (REAL, C=220)  (C=220 candidates, N=96 prompts, K=12)
full greedy: value +2.128, cells 21120 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 8 | +1.668 | 78% | 1.2 | 1760 | 8% |
| subsample | 16 | +1.809 | 85% | 1.2 | 3520 | 17% |
| subsample | 24 | +1.874 | 88% | 3.2 | 5280 | 25% |
| subsample | 48 | +2.024 | 95% | 6.0 | 10560 | 50% |
| subsample+stochastic | 8 | +1.667 | 78% | 0.8 | 1640 | 8% |
| subsample+stochastic | 16 | +1.824 | 86% | 2.5 | 3276 | 16% |
| subsample+stochastic | 24 | +1.899 | 89% | 4.0 | 4866 | 23% |
| subsample+stochastic | 48 | +1.943 | 91% | 4.2 | 9864 | 47% |

### synthetic (C=100)  (C=100 candidates, N=300 prompts, K=12)
full greedy: value +2.002, cells 30000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 16 | +1.889 | 94% | 9.2 | 1600 | 5% |
| subsample | 32 | +1.970 | 98% | 11.2 | 3200 | 11% |
| subsample | 64 | +2.002 | 100% | 12.0 | 6400 | 21% |
| subsample | 128 | +2.002 | 100% | 12.0 | 12800 | 43% |
| subsample+stochastic | 16 | +1.837 | 92% | 8.5 | 1508 | 5% |
| subsample+stochastic | 32 | +1.885 | 94% | 9.2 | 3008 | 10% |
| subsample+stochastic | 64 | +1.884 | 94% | 9.2 | 5904 | 20% |
| subsample+stochastic | 128 | +1.841 | 92% | 8.8 | 12000 | 40% |

### synthetic (C=200)  (C=200 candidates, N=300 prompts, K=12)
full greedy: value +2.022, cells 60000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 16 | +1.843 | 91% | 8.5 | 3200 | 5% |
| subsample | 32 | +1.989 | 98% | 11.2 | 6400 | 11% |
| subsample | 64 | +2.002 | 99% | 11.5 | 12800 | 21% |
| subsample | 128 | +2.015 | 100% | 11.8 | 25600 | 43% |
| subsample+stochastic | 16 | +1.772 | 88% | 7.5 | 2976 | 5% |
| subsample+stochastic | 32 | +1.861 | 92% | 8.8 | 5944 | 10% |
| subsample+stochastic | 64 | +1.844 | 91% | 8.2 | 11952 | 20% |
| subsample+stochastic | 128 | +1.889 | 93% | 9.0 | 23840 | 40% |
