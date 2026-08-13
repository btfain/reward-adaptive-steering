# S1 (revised) — procedural-prompt basis + learned router

Qwen2.5-7B-Instruct. X = 36 curated candidate moves (subproc.1 abstracted). Swing(x,p)=RM(sys=p)−RM(base), base reused from results/steer_rm_7b. Greedy submodular basis (subproc.2), K=8; router read-layer swept [8, 14, 18, 22, 27], PCA-40 (subproc.3). n_train=150, n_test=50, m=4. RM1=Skywork-Reward-V2-Qwen3-0.6B.

References (A2/7B): prompting +1.08, best-of-n +1.40, contrastive-steering ~0. Base distinct-2 0.576.

## Subprocedure 2 — greedy submodular basis (value-vs-K, oracle per-prompt captured swing)
| K | marginal | mean captured swing | move |
|---|---|---|---|
| 1 | +1.258 | +1.258 | Be concise and remove any filler, repetition, or throat-clearing. |
| 2 | +0.432 | +1.691 | Be direct and give a helpful answer rather than deflecting or over-hedging. |
| 3 | +0.121 | +1.812 | Be honest about what is unknown or unknowable rather than fabricating an answer. |
| 4 | +0.087 | +1.899 | Answer the question directly in the first sentence, then provide supporting deta |
| 5 | +0.058 | +1.958 | Prioritize factual accuracy and avoid presenting speculation as fact. |
| 6 | +0.051 | +2.009 | Organize your response with clear structure, using short paragraphs or bullet po |
| 7 | +0.042 | +2.051 | If you are uncertain about anything, say so explicitly and explain the source of |
| 8 | +0.027 | +2.078 | Acknowledge the user's underlying goal and address that, not just the literal qu |

## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM)
| layer | arm | acc train | acc test | ΔRM1 (realized) |
|---|---|---|---|---|
| 8 | linear | 0.97 | 0.32 | +0.860  ⟵ best |
| 8 | mlp | 1.00 | 0.26 | +0.959 |
| 14 | linear | 1.00 | 0.16 | +0.673 |
| 14 | mlp | 1.00 | 0.32 | +1.054 |
| 18 | linear | 0.97 | 0.22 | +0.730 |
| 18 | mlp | 1.00 | 0.28 | +0.940 |
| 22 | linear | 1.00 | 0.14 | +0.716 |
| 22 | mlp | 1.00 | 0.28 | +0.888 |
| 27 | linear | 1.00 | 0.18 | +0.655 |
| 27 | mlp | 0.99 | 0.30 | +1.008 |

## Held-out result (best router: layer 8, linear)
- **Learned router ΔRM1: +0.860 [+0.431, +1.309]** (routes to 'none' on 2% of prompts).
- Best SINGLE unconditional move (k=1): +0.905 [+0.372, +1.423]  — the load-bearing baseline; router must beat this for conditioning to be worth it.
- Oracle-over-basis ceiling: +1.986 [+1.618, +2.362].
- Router-routed generations distinct-2 0.443 (base 0.576); guard for collapse.

## Reading
- **router > best-single-move (CIs)** ⇒ conditioning pays — the prompt-basis method works where steering's did not. Compare the gap to the oracle-over-basis ceiling (routing quality).
- **router ≈ best-single-move** ⇒ no conditioning value yet: either the routing signal is weak or X lacks type-specific moves (⇒ build the LLM-from-preferences generator for a stronger X).
- **large train→test accuracy gap** ⇒ router is data-limited at n_train — scale prompts (only grows the parallel swing precompute, no redesign).
- Judge magnitudes against prompting (+1.08) and best-of-n (+1.40).
