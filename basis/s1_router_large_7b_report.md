# Router study (offline) — large_7b: value-regression + early-stopping + honest selection

Cached states /home/users/btf12/reward-adaptive-steering/results/prompt_basis_large_7b/states.npz + train swings. 450 valid prompts (270 router-train / 90 val / 90 eval), K=8 moves, PCA-40, layers [8, 14, 18, 22, 27]. Selected on VAL realized-ΔRM, reported on held-out EVAL (train swings, m_swing — noisier than the run's m_test de-biased oracle ~+0.81).

Eval baselines: single move +0.399 [+0.180, +0.637]; naive(biased) oracle +1.108 [+0.919, +1.320].

| variant | cap | best layer | val ΔRM | **eval ΔRM** |
|---|---|---|---|---|
| cls | linear | 8 | +0.297 | **+0.069** |
| cls | mlp | 18 | +0.369 | **+0.271** |
| reg | linear | 22 | +0.431 | **+0.180** |
| reg | mlp | 22 | +0.465 | **+0.227** |

## Best (val-selected): reg/mlp @ layer 22  hp={'hidden': 64, 'dropout': 0.5, 'lr': 0.05, 'wd': 0.1, 'epochs': 800, 'patience': 40}
- **eval ΔRM +0.227 [-0.057, +0.518]** vs single move +0.399  (below single).

## Reading
- **reg beats cls and clears single (eval CI vs single)** ⇒ conditioning IS extractable — the hard argmax classifier / overfitting was the problem, not the signal. Confirm on the honest test set.
- **all variants ≈ single** ⇒ h→move is genuinely hard here ⇒ clean Subproject-1 negative on extraction; carry routing to multi-turn (where the 'which move' signal should be far stronger).
