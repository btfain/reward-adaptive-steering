# S1 (revised) — procedural-prompt basis + router (DETRUNCATED, de-biased)

Qwen2.5-7B-Instruct. X = 20 curated moves. Swing(x,p)=RM(sys=p)−RM(base), base regenerated at max_new_tokens=512 (truncation now 32% on move gens; was 72% at 128). Greedy submodular basis K=8; router read-layer swept [8, 14, 18, 22, 27], PCA-40. n_train=120, n_test=50, m_swing=4, m_test=12 (val/test split). RM1=Skywork-Reward-V2-Qwen3-0.6B.

References (A2/7B, TRUNCATED — now suspect): prompting +1.08, best-of-n +1.40, steering ~0. Base distinct-2 0.630.

## Subprocedure 2 — greedy submodular basis (value-vs-K; does 'concise' still dominate post-detrunc?)
| K | marginal | mean captured swing | move |
|---|---|---|---|
| 1 | +0.520 | +0.520 | Explain the reasoning behind your answer, not just the conclusion. |
| 2 | +0.252 | +0.772 | Prioritize factual accuracy and avoid presenting speculation as fact. |
| 3 | +0.139 | +0.912 | Where the question is ambiguous, state your interpretation explicitly and answer |
| 4 | +0.115 | +1.027 | Support your claims with specific data, numbers, or evidence where possible. |
| 5 | +0.072 | +1.099 | Be concise and remove any filler, repetition, or throat-clearing. |
| 6 | +0.064 | +1.163 | If code is involved, provide complete, runnable code with brief explanatory comm |
| 7 | +0.058 | +1.221 | Be direct and give a helpful answer rather than deflecting or over-hedging. |
| 8 | +0.039 | +1.259 | If you are uncertain about anything, say so explicitly and explain the source of |

## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM, test half)
| layer | arm | acc train | acc test | ΔRM1 |
|---|---|---|---|---|
| 8 | linear | 1.00 | 0.26 | +0.104 |
| 8 | mlp | 1.00 | 0.18 | +0.151 |
| 14 | linear | 1.00 | 0.20 | +0.033 |
| 14 | mlp | 1.00 | 0.28 | +0.325  ⟵ best |
| 18 | linear | 1.00 | 0.20 | -0.133 |
| 18 | mlp | 1.00 | 0.22 | +0.260 |
| 22 | linear | 1.00 | 0.18 | +0.018 |
| 22 | mlp | 1.00 | 0.12 | +0.069 |
| 27 | linear | 1.00 | 0.10 | +0.028 |
| 27 | mlp | 0.99 | 0.16 | -0.013 |

## Held-out result (best router: layer 14, mlp; all scored on the test half)
- **Learned router ΔRM1: +0.325 [+0.060, +0.578]**.
- Best SINGLE unconditional move (k=1): -0.012 [-0.320, +0.278]  — the load-bearing baseline; router must beat this for conditioning to pay.
- **De-biased oracle-over-basis (pick on val, score on test): +0.782 [+0.538, +1.057]** — the TRUE routing ceiling; its gap over the single move is the real conditioning headroom.
- Naive (biased) oracle for reference: +1.038 [+0.799, +1.294] — inflation vs de-biased = the winner's-curse we removed.
- Routed generations distinct-2 0.318 (base 0.630).

## Reading
- **truncation now low** confirms the fix; compare the surviving swings to the 128-token run to see how much single-turn 'headroom' was truncation-avoidance.
- **de-biased oracle ≫ best-single-move** ⇒ real single-turn conditioning headroom exists (then judge whether the router captures it: router vs single). **de-biased ≈ single** ⇒ conditioning is mostly illusory here ⇒ carry routing to multi-turn (Subproject 2).
- **router > single (CIs)** ⇒ conditioning pays and is learnable now; **router ≈ single** with a large de-biased gap ⇒ signal exists but router is data-limited (scale prompts).
- Judge magnitudes against the detruncated single move; the old +1.08/+1.40 refs are truncation-suspect.
