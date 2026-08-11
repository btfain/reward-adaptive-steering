# S1.2 — conditional steering controller (type-dependent positive control)

Types = sign of the read state's top-PC projection (recoverable by construction); A→hedge+, B→hedge−. Qwen2.5-7B-Instruct, steer L18, read L18, rank 2, soft mag cap 40 (unbounded coeffs, penalty-shaped). n=6 pool. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): held-out **94%** (~100% by construction ⇒ a routing failure is purely optimization).

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.049 [-0.067, +0.194] | +0.068 | +0.012 |
| linear | -0.029 [-0.195, +0.136] | -0.078 | +0.070 |
| mlp | -0.026 [-0.117, +0.065] | -0.000 | -0.077 |

## Routing — cosine between mean injection direction for type A vs B (≈+1 = same direction = no routing; ≤0 = opposite = routing)
- **global**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-2.64, 3.09], a|B=[-2.64, 3.09])
- **linear**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[58.19, -64.65], a|B=[80.01, -93.75])
- **mlp**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-5.76, 11.1], a|B=[-5.76, 11.1])

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants hedge↓
- **global** type A: Δwords +0.80, Δhedge +0.07, Δquestions -0.04
- **global** type B: Δwords -0.80, Δhedge -0.01, Δquestions +0.04
- **linear** type A: Δwords +10.22, Δhedge -0.08, Δquestions -0.13
- **linear** type B: Δwords +2.64, Δhedge -0.07, Δquestions +0.05
- **mlp** type A: Δwords -0.43, Δhedge -0.00, Δquestions +0.09
- **mlp** type B: Δwords -0.55, Δhedge +0.08, Δquestions +0.02

## Reading
Conditioning value = best conditional Δ-R (-0.026, mlp) − global Δ-R (+0.049) = **-0.075**.
Routing cos(δ̄A,δ̄B): linear +1.00, mlp +1.00 (< 0.5 = genuine per-type routing, not one stronger global vector).
**S1.2 NOT green**: requires a conditional arm routing to different directions per type (cos < 0.5) AND Δ-R > global AND a legible type signal (probe > 75%). Not met — an arm that beats global without directional routing is just one stronger GLOBAL vector (prompt-distribution artifact); a low probe would mean the cue is washed out.
