# S1.2 — conditional steering controller (type-dependent positive control)

Types = sign of the read state's top-PC projection (recoverable by construction); A→hedge+, B→hedge−. SmolLM2-1.7B-Instruct, steer L16, read L16, rank 2, soft mag cap 117 (unbounded coeffs, penalty-shaped). n=12 pool. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): held-out **90%** (~100% by construction ⇒ a routing failure is purely optimization).

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.042 [-0.071, +0.157] | +0.043 | +0.042 |
| linear | +0.035 [-0.081, +0.156] | +0.105 | -0.072 |
| mlp | +0.057 [-0.025, +0.142] | +0.007 | +0.134 |

## Routing — cosine between mean injection direction for type A vs B (≈+1 = same direction = no routing; ≤0 = opposite = routing)
- **global**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[23.99, -15.81], a|B=[23.99, -15.81])
- **linear**: cos(δ̄A, δ̄B) = **+0.82**  (a|A=[117.82, -143.65], a|B=[14.91, -45.1])
- **mlp**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-67.3, -17.07], a|B=[-67.6, -17.25])

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants hedge↓
- **global** type A: Δwords -1.73, Δhedge +0.04, Δquestions +0.04
- **global** type B: Δwords -0.50, Δhedge -0.03, Δquestions +0.01
- **linear** type A: Δwords -1.77, Δhedge +0.09, Δquestions -0.01
- **linear** type B: Δwords -0.71, Δhedge +0.06, Δquestions -0.02
- **mlp** type A: Δwords -1.43, Δhedge +0.01, Δquestions +0.10
- **mlp** type B: Δwords -0.64, Δhedge -0.11, Δquestions -0.03

## Reading
Conditioning value = best conditional Δ-R (+0.057, mlp) − global Δ-R (+0.042) = **+0.015**.
Routing cos(δ̄A,δ̄B): linear +0.82, mlp +1.00 (< 0.5 = genuine per-type routing, not one stronger global vector).
**S1.2 NOT green**: requires a conditional arm routing to different directions per type (cos < 0.5) AND Δ-R > global AND a legible type signal (probe > 75%). Not met — an arm that beats global without directional routing is just one stronger GLOBAL vector (prompt-distribution artifact); a low probe would mean the cue is washed out.
