# S1 (revised) — procedural-prompt basis + router (DETRUNCATED, de-biased)

Qwen2.5-7B-Instruct. X = 20 curated moves. Swing(x,p)=RM(sys=p)−RM(base), base regenerated at max_new_tokens=768 (truncation now 19% on move gens; was 72% at 128). Greedy submodular basis K=8; router read-layer swept [8, 14, 18, 22, 27], PCA-40. n_train=450, n_test=150, m_swing=4, m_test=12 (val/test split). RM1=Skywork-Reward-V2-Qwen3-0.6B.

References (A2/7B, TRUNCATED — now suspect): prompting +1.08, best-of-n +1.40, steering ~0. Base distinct-2 0.540.

## Subprocedure 2 — greedy submodular basis (value-vs-K; does 'concise' still dominate post-detrunc?)
| K | marginal | mean captured swing | move |
|---|---|---|---|
| 1 | +0.585 | +0.585 | Give a thorough, complete answer that addresses every part of the question. |
| 2 | +0.220 | +0.805 | Work through the problem step by step, showing your reasoning before stating the |
| 3 | +0.130 | +0.935 | Explain the reasoning behind your answer, not just the conclusion. |
| 4 | +0.083 | +1.018 | Support your claims with specific data, numbers, or evidence where possible. |
| 5 | +0.062 | +1.080 | If code is involved, provide complete, runnable code with brief explanatory comm |
| 6 | +0.036 | +1.116 | Provide an in-depth, expert-level explanation with appropriate technical depth. |
| 7 | +0.033 | +1.149 | Be direct and give a helpful answer rather than deflecting or over-hedging. |
| 8 | +0.027 | +1.176 | Include a concrete example to illustrate your main point. |

## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM, test half)
| layer | arm | acc train | acc test | ΔRM1 |
|---|---|---|---|---|
| 8 | linear | 0.46 | 0.17 | +0.196 |
| 8 | mlp | 0.71 | 0.19 | +0.195 |
| 14 | linear | 0.45 | 0.15 | +0.243 |
| 14 | mlp | 1.00 | 0.19 | +0.283 |
| 18 | linear | 0.43 | 0.23 | +0.360  ⟵ best |
| 18 | mlp | 1.00 | 0.17 | +0.406 |
| 22 | linear | 0.45 | 0.19 | +0.368 |
| 22 | mlp | 1.00 | 0.15 | +0.305 |
| 27 | linear | 0.41 | 0.16 | +0.379 |
| 27 | mlp | 1.00 | 0.20 | +0.340 |

## Held-out result (best router: layer 18, linear; all scored on the test half)
- **Learned router ΔRM1: +0.360 [+0.154, +0.568]**.
- Best SINGLE unconditional move (k=1): +0.355 [+0.164, +0.545]  — the load-bearing baseline; router must beat this for conditioning to pay.
- **De-biased oracle-over-basis (pick on val, score on test): +0.813 [+0.661, +0.974]** — the TRUE routing ceiling; its gap over the single move is the real conditioning headroom.
- Naive (biased) oracle for reference: +1.108 [+0.943, +1.277] — inflation vs de-biased = the winner's-curse we removed.
- Routed generations distinct-2 0.261 (base 0.540).

## Reading
- **truncation now low** confirms the fix; compare the surviving swings to the 128-token run to see how much single-turn 'headroom' was truncation-avoidance.
- **de-biased oracle ≫ best-single-move** ⇒ real single-turn conditioning headroom exists (then judge whether the router captures it: router vs single). **de-biased ≈ single** ⇒ conditioning is mostly illusory here ⇒ carry routing to multi-turn (Subproject 2).
- **router > single (CIs)** ⇒ conditioning pays and is learnable now; **router ≈ single** with a large de-biased gap ⇒ signal exists but router is data-limited (scale prompts).
- Judge magnitudes against the detruncated single move; the old +1.08/+1.40 refs are truncation-suspect.
