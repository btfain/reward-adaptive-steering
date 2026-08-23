# B1 online bandit router — b1_1500: FROZEN-head vs FINE-TUNED-encoder (roberta-base)

Controlled single-turn test at n_train=1500 on the fixed 8-move basis. Online hardened REINFORCE, reward = RM(move) - base_ref, decline arm = 0. Router-eval on 600 held-out prompts (1 gen each). single +0.351 (0.205, 0.495); de-biased oracle +0.918 (0.728, 1.141) (n=100).

| arm | eval ΔRM | vs single | vs single (paired) | trainable params | final entropy |
|---|---|---|---|---|---|
| frozen | +0.349 (0.209, 0.49) | -0.002 | -0.002 [-0.147, +0.141] | 7,690 | 1.592 |
| finetune | +0.351 (0.222, 0.488) | -0.000 | -0.000 [-0.135, +0.133] | 124,653,322 | 1.439 |

## Headline — fine-tune vs frozen (paired, same eval prompts)
- fine-tune − frozen = +0.002 [-0.132, +0.146]; fine-tune eval +0.351.
- clears the ~+0.40 single-turn ceiling: no.

## Anti-collapse guard (rule 4)
- frozen: final move-usage [0.089, 0.474, 0.083, 0.12, 0.169, 0.011, 0.024, 0.007, 0.023] (arm0=decline); entropy 1.592 ⇒ OK (spread).
- finetune: final move-usage [0.247, 0.322, 0.167, 0.155, 0.082, 0.007, 0.007, 0.004, 0.01] (arm0=decline); entropy 1.439 ⇒ OK (spread).

## Verdict
- fine-tune ≈/below frozen (paired CI includes 0) ⇒ the one untested lever does NOT exceed the frozen ceiling ⇒ single-turn conditioning BOUND confirmed ⇒ pivot to Study 2 (bandit carries over).

## Cost
- frozen: 3.12 GPU-h train, 7,690 trainable params
- finetune: 11.86 GPU-h train, 124,653,322 trainable params
