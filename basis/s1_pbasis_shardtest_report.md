# S1 (revised) — procedural-prompt basis + router (DETRUNCATED, de-biased)

Qwen2.5-7B-Instruct. X = 2 curated moves. Swing(x,p)=RM(sys=p)−RM(base), base regenerated at max_new_tokens=64 (truncation now 75% on move gens; was 72% at 128). Greedy submodular basis K=2; router read-layer swept [8, 18], PCA-4. n_train=6, n_test=3, m_swing=2, m_test=2 (val/test split). RM1=Skywork-Reward-V2-Qwen3-0.6B.

References (A2/7B, TRUNCATED — now suspect): prompting +1.08, best-of-n +1.40, steering ~0. Base distinct-2 0.869.

## Subprocedure 2 — greedy submodular basis (value-vs-K; does 'concise' still dominate post-detrunc?)
| K | marginal | mean captured swing | move |
|---|---|---|---|
| 1 | +1.934 | +1.934 | Be concise and remove any filler, repetition, or throat-clearing. |
| 2 | +0.000 | +1.934 | Give a thorough, complete answer that addresses every part of the question. |

## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM, test half)
| layer | arm | acc train | acc test | ΔRM1 |
|---|---|---|---|---|
| 8 | linear | 1.00 | 0.33 | +0.475  ⟵ best |
| 8 | mlp | 1.00 | 0.33 | +0.475 |
| 18 | linear | 1.00 | 0.33 | +0.475 |
| 18 | mlp | 1.00 | 0.33 | +0.475 |

## Held-out result (best router: layer 8, linear; all scored on the test half)
- **Learned router ΔRM1: +0.475 [-1.426, +1.984]**.
- Best SINGLE unconditional move (k=1): +0.475 [-1.426, +1.984]  — the load-bearing baseline; router must beat this for conditioning to pay.
- **De-biased oracle-over-basis (pick on val, score on test): +0.438 [+0.000, +0.867]** — the TRUE routing ceiling; its gap over the single move is the real conditioning headroom.
- Naive (biased) oracle for reference: +0.951 [+0.000, +1.984] — inflation vs de-biased = the winner's-curse we removed.
- Routed generations distinct-2 0.726 (base 0.869).

## Reading
- **truncation now low** confirms the fix; compare the surviving swings to the 128-token run to see how much single-turn 'headroom' was truncation-avoidance.
- **de-biased oracle ≫ best-single-move** ⇒ real single-turn conditioning headroom exists (then judge whether the router captures it: router vs single). **de-biased ≈ single** ⇒ conditioning is mostly illusory here ⇒ carry routing to multi-turn (Subproject 2).
- **router > single (CIs)** ⇒ conditioning pays and is learnable now; **router ≈ single** with a large de-biased gap ⇒ signal exists but router is data-limited (scale prompts).
- Judge magnitudes against the detruncated single move; the old +1.08/+1.40 refs are truncation-suspect.
