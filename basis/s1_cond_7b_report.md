# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→hedge− (z-scored φ). SmolLM2-1.7B, steer L18, read L18, rank 2, magnitude FIXED at cap 35 (direction-only routing; coeffs a∈[−1,1]). n=6 pool. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): train 100%, held-out **54%** — near 100% ⇒ routing signal present (failure would be optimization/reward); ~50% ⇒ cue washed out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.024 [-0.111, +0.180] | +0.238 | -0.190 |
| linear | +0.045 [-0.079, +0.192] | +0.020 | +0.070 |
| mlp | -0.044 [-0.202, +0.122] | +0.198 | -0.285 |

## Routing — cosine between mean injection direction for type A vs B (≈+1 = same direction = no routing; ≤0 = opposite = routing)
- **global**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-2.11, 2.57], a|B=[-2.11, 2.57])
- **linear**: cos(δ̄A, δ̄B) = **+0.96**  (a|A=[42.37, -51.72], a|B=[52.81, -47.63])
- **mlp**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-6.39, 11.39], a|B=[-6.39, 11.39])

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +0.02, Δhedge +0.30, Δquestions +0.04
- **global** type B: Δwords +0.28, Δhedge +0.24, Δquestions -0.04
- **linear** type A: Δwords -2.02, Δhedge +0.02, Δquestions +0.04
- **linear** type B: Δwords -1.27, Δhedge -0.09, Δquestions -0.14
- **mlp** type A: Δwords +0.66, Δhedge +0.25, Δquestions +0.08
- **mlp** type B: Δwords -1.00, Δhedge +0.36, Δquestions +0.28

## Reading
Conditioning value = best conditional Δ-R (+0.045, linear) − global Δ-R (+0.024) = **+0.021**.
Routing cos(δ̄A,δ̄B): linear +0.96, mlp +1.00 (< 0.5 = genuine per-type routing, not one stronger global vector).
**S1.2 NOT green**: requires a conditional arm routing to different directions per type (cos < 0.5) AND Δ-R > global AND a legible type signal (probe > 75%). Not met — an arm that beats global without directional routing is just one stronger GLOBAL vector (prompt-distribution artifact); a low probe would mean the cue is washed out.
