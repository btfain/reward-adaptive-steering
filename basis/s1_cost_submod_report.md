# P1(ii) cost-aware submodular selection — offline validation

Cost-aware greedy (sampled marginals + successive elimination) vs full greedy on a generation-cost oracle. Metric: basis overlap + value ratio, and fraction of C×N cells queried.

### large_7b (real, C=20)  (C=20 candidates, N=450 prompts, K=8)
full greedy: value +1.176, cells 9000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 8 | +1.076 | 91% | 4.0 | 160 | 2% |
| subsample | 16 | +1.083 | 92% | 4.8 | 320 | 4% |
| subsample | 32 | +1.121 | 95% | 5.0 | 640 | 7% |
| subsample | 64 | +1.141 | 97% | 5.8 | 1280 | 14% |
| subsample | 128 | +1.165 | 99% | 6.0 | 2560 | 28% |
| subsample+stochastic | 8 | +1.056 | 90% | 4.0 | 152 | 2% |
| subsample+stochastic | 16 | +1.065 | 91% | 4.0 | 308 | 3% |
| subsample+stochastic | 32 | +1.113 | 95% | 4.5 | 624 | 7% |
| subsample+stochastic | 64 | +1.099 | 93% | 4.8 | 1248 | 14% |
| subsample+stochastic | 128 | +1.133 | 96% | 5.5 | 2496 | 28% |

### synthetic (C=100)  (C=100 candidates, N=300 prompts, K=8)
full greedy: value +1.870, cells 30000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 16 | +1.822 | 97% | 6.2 | 1600 | 5% |
| subsample | 32 | +1.799 | 96% | 5.0 | 3200 | 11% |
| subsample | 64 | +1.822 | 97% | 6.2 | 6400 | 21% |
| subsample | 128 | +1.848 | 99% | 7.0 | 12800 | 43% |
| subsample+stochastic | 16 | +1.772 | 95% | 6.0 | 1528 | 5% |
| subsample+stochastic | 32 | +1.706 | 91% | 4.5 | 3008 | 10% |
| subsample+stochastic | 64 | +1.819 | 97% | 6.2 | 6016 | 20% |
| subsample+stochastic | 128 | +1.772 | 95% | 5.8 | 11904 | 40% |

### synthetic (C=200)  (C=200 candidates, N=300 prompts, K=8)
full greedy: value +1.869, cells 60000 (=C×N)
| method | s (prompts) | value | % of full | overlap /K | cells | % of full |
|---|---|---|---|---|---|---|
| subsample | 16 | +1.750 | 94% | 4.8 | 3200 | 5% |
| subsample | 32 | +1.811 | 97% | 5.8 | 6400 | 11% |
| subsample | 64 | +1.840 | 98% | 6.5 | 12800 | 21% |
| subsample | 128 | +1.848 | 99% | 6.8 | 25600 | 43% |
| subsample+stochastic | 16 | +1.672 | 89% | 4.2 | 3052 | 5% |
| subsample+stochastic | 32 | +1.756 | 94% | 5.2 | 6072 | 10% |
| subsample+stochastic | 64 | +1.772 | 95% | 5.8 | 12048 | 20% |
| subsample+stochastic | 128 | +1.828 | 98% | 6.2 | 23872 | 40% |
