# B0.5 learning curve — exact-policy ceiling vs train size (full-dim, 12 seeds)

Is the ~+0.38 frozen ceiling data-limited (rises with prompts ⇒ B1 scaling helps) or information-limited (flat ⇒ signal not in the text ⇒ pivot)? single +0.300, oracle ≈ +0.78.

| encoder | n_tr≈67 | n_tr≈135 | n_tr≈202 | n_tr≈270 |
|---|---|---|---|---|
| distilroberta-base | +0.315 | +0.325 | +0.353 | +0.361 |
| intfloat/e5-large-v2 | +0.304 | +0.358 | +0.352 | +0.359 |

## Reading
- Δ(full−quarter) per encoder: distilroberta-base +0.047, intfloat/e5-large-v2 +0.055.
- **Still climbing at n_tr=270** ⇒ data-limited ⇒ B1 (more prompts) can lift the ceiling — scale.
- **Flat by n_tr=270** ⇒ information-limited ⇒ the prompt text lacks the best-move signal ⇒ more prompts won't help; carry the working bandit to Study 2 (multi-turn state carries richer signal).
