# Encoder-as-steering-controller probe (offline, free) — steer_reach_7b

Regress features → per-prompt optimal δ (deltas.npz, 200×3584), ridge λ=1.0, held-out 50 test prompts. Metric = direction recovery (mean cosine) + aggregate R². Baseline = the LLM's own read state h(x) (steer_reach's 'not controllable' finding).

| features | dim | held-out cosine(pred,true δ) | held-out R² |
|---|---|---|---|
| h(x) LLM read state | 3584 | +0.115 | -0.597 |
| roberta-base | 768 | +0.088 | -0.909 |
| intfloat/e5-large-v2 | 1024 | +0.107 | -0.529 |

## Reading
- intfloat/e5-large-v2 cosine +0.107 > 0 but ≈ h (+0.115) ⇒ modest, not clearly better than the LLM state ⇒ weak case; revisit only if B1 also motivates single-turn.
- Direction (cosine) is the controllability signal — magnitude is capped in steering; R² over 3584 dims is naturally low, so read the h-vs-encoder DELTA, not the absolute.
