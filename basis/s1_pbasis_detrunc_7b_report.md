# S1 (revised) — procedural-prompt basis + router (DETRUNCATED, de-biased)

Qwen2.5-7B-Instruct. X = 20 curated moves. Swing(x,p)=RM(sys=p)−RM(base), base regenerated at max_new_tokens=256 (truncation now 57% on move gens; was 72% at 128). Greedy submodular basis K=8; router read-layer swept [8, 14, 18, 22, 27], PCA-40. n_train=120, n_test=50, m_swing=4, m_test=12 (val/test split). RM1=Skywork-Reward-V2-Qwen3-0.6B.

References (A2/7B, TRUNCATED — now suspect): prompting +1.08, best-of-n +1.40, steering ~0. Base distinct-2 0.653.

## Subprocedure 2 — greedy submodular basis (value-vs-K; does 'concise' still dominate post-detrunc?)
| K | marginal | mean captured swing | move |
|---|---|---|---|
| 1 | +0.854 | +0.854 | Be direct and give a helpful answer rather than deflecting or over-hedging. |
| 2 | +0.444 | +1.298 | Be concise and remove any filler, repetition, or throat-clearing. |
| 3 | +0.203 | +1.501 | If you are uncertain about anything, say so explicitly and explain the source of |
| 4 | +0.106 | +1.608 | Organize your response with clear structure, using short paragraphs or bullet po |
| 5 | +0.064 | +1.671 | Prioritize factual accuracy and avoid presenting speculation as fact. |
| 6 | +0.046 | +1.718 | Explain the reasoning behind your answer, not just the conclusion. |
| 7 | +0.024 | +1.741 | Briefly define any technical terms you introduce. |
| 8 | +0.016 | +1.757 | Work through the problem step by step, showing your reasoning before stating the |

## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM, test half)
| layer | arm | acc train | acc test | ΔRM1 |
|---|---|---|---|---|
| 8 | linear | 0.98 | 0.26 | +1.060 |
| 8 | mlp | 1.00 | 0.32 | +1.185  ⟵ best |
| 14 | linear | 1.00 | 0.30 | +0.856 |
| 14 | mlp | 1.00 | 0.22 | +0.884 |
| 18 | linear | 1.00 | 0.32 | +0.874 |
| 18 | mlp | 1.00 | 0.24 | +0.952 |
| 22 | linear | 1.00 | 0.26 | +0.752 |
| 22 | mlp | 1.00 | 0.24 | +0.861 |
| 27 | linear | 1.00 | 0.28 | +0.927 |
| 27 | mlp | 1.00 | 0.28 | +0.852 |

## Held-out result (best router: layer 8, mlp; all scored on the test half)
- **Learned router ΔRM1: +1.185 [+0.652, +1.712]**.
- Best SINGLE unconditional move (k=1): +0.719 [+0.333, +1.108]  — the load-bearing baseline; router must beat this for conditioning to pay.
- **De-biased oracle-over-basis (pick on val, score on test): +1.904 [+1.485, +2.360]** — the TRUE routing ceiling; its gap over the single move is the real conditioning headroom.
- Naive (biased) oracle for reference: +2.006 [+1.601, +2.446] — inflation vs de-biased = the winner's-curse we removed.
- Routed generations distinct-2 0.330 (base 0.653).

## Reading
- **truncation now low** confirms the fix; compare the surviving swings to the 128-token run to see how much single-turn 'headroom' was truncation-avoidance.
- **de-biased oracle ≫ best-single-move** ⇒ real single-turn conditioning headroom exists (then judge whether the router captures it: router vs single). **de-biased ≈ single** ⇒ conditioning is mostly illusory here ⇒ carry routing to multi-turn (Subproject 2).
- **router > single (CIs)** ⇒ conditioning pays and is learnable now; **router ≈ single** with a large de-biased gap ⇒ signal exists but router is data-limited (scale prompts).
- Judge magnitudes against the detruncated single move; the old +1.08/+1.40 refs are truncation-suspect.
