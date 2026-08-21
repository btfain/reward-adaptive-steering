# Exploration sweep (offline, free) — large_7b: entropy_beta x lr to STOP collapse

Sampled REINFORCE on cached enc features + swings, m=1 noise σ=1.0, 8 seeds, 8 epochs (matches online). single move +0.269; exact-policy ceiling ≈ +0.38. Collapse = max move-usage; want eval near ceiling AND max-usage < ~0.7 (still conditioning).

| head | beta | lr | eval ΔRM | max-usage | entropy | note |
|---|---|---|---|---|---|---|
| linear~frozen | 0.02 | 0.02 | +0.291 | 0.20 | 1.94 | weak |
| linear~frozen | 0.02 | 0.05 | +0.335 | 0.31 | 1.57 | ok |
| linear~frozen | 0.05 | 0.02 | +0.313 | 0.19 | 1.96 | ok |
| linear~frozen | 0.05 | 0.05 | +0.366 | 0.29 | 1.64 | ok |
| linear~frozen | 0.1 | 0.02 | +0.319 | 0.18 | 1.98 | ok |
| linear~frozen | 0.1 | 0.05 | +0.363 | 0.27 | 1.76 | ok |
| linear~frozen | 0.2 | 0.02 | +0.349 | 0.18 | 2.03 | ok |
| linear~frozen | 0.2 | 0.05 | +0.319 | 0.23 | 1.86 | ok |
| linear~frozen | 0.4 | 0.02 | +0.306 | 0.18 | 2.08 | ok |
| linear~frozen | 0.4 | 0.05 | +0.284 | 0.19 | 2.00 | weak |
| mlp~finetune | 0.02 | 0.02 | +0.294 | 0.69 | 0.73 | weak |
| mlp~finetune | 0.02 | 0.05 | +0.235 | 0.68 | 0.65 | weak |
| mlp~finetune | 0.05 | 0.02 | +0.325 | 0.59 | 1.03 | ok |
| mlp~finetune | 0.05 | 0.05 | +0.301 | 0.47 | 1.10 | ok |
| mlp~finetune | 0.1 | 0.02 | +0.362 | 0.42 | 1.43 | ok |
| mlp~finetune | 0.1 | 0.05 | +0.282 | 0.54 | 1.15 | weak |
| mlp~finetune | 0.2 | 0.02 | +0.309 | 0.29 | 1.84 | ok |
| mlp~finetune | 0.2 | 0.05 | +0.275 | 0.34 | 1.67 | weak |
| mlp~finetune | 0.4 | 0.02 | +0.299 | 0.20 | 2.04 | ok |
| mlp~finetune | 0.4 | 0.05 | +0.261 | 0.21 | 1.99 | weak |

## Pick (MLP≈fine-tune arm, not collapsed, best eval)
- **entropy_beta=0.1, lr=0.02** → eval +0.362, max-usage 0.42 ⇒ use these online (and lower lr_enc so the encoder can't drive collapse either).
