# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→hedge− (z-scored φ). SmolLM2-1.7B, steer L16, read L16, rank 2, magnitude FIXED at cap 106 (direction-only routing; coeffs a∈[−1,1]). n=12 pool. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): train 100%, held-out **56%** — near 100% ⇒ routing signal present (failure would be optimization/reward); ~50% ⇒ cue washed out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.032 [-0.082, +0.155] | -0.058 | +0.122 |
| linear | +0.045 [-0.051, +0.140] | +0.003 | +0.087 |
| mlp | +0.009 [-0.111, +0.130] | -0.090 | +0.109 |

## Routing — cosine between mean injection direction for type A vs B (≈+1 = same direction = no routing; ≤0 = opposite = routing)
- **global**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[22.64, -15.02], a|B=[22.64, -15.02])
- **linear**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[38.37, -78.21], a|B=[44.99, -83.0])
- **mlp**: cos(δ̄A, δ̄B) = **+1.00**  (a|A=[-61.27, -15.64], a|B=[-61.55, -16.17])

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +1.23, Δhedge -0.05, Δquestions +0.12
- **global** type B: Δwords +0.28, Δhedge -0.10, Δquestions -0.03
- **linear** type A: Δwords +0.53, Δhedge +0.00, Δquestions -0.03
- **linear** type B: Δwords +0.37, Δhedge -0.07, Δquestions -0.00
- **mlp** type A: Δwords +0.69, Δhedge -0.08, Δquestions +0.02
- **mlp** type B: Δwords +2.88, Δhedge -0.09, Δquestions -0.03

## Reading
Conditioning value = best conditional Δ-R (+0.045, linear) − global Δ-R (+0.032) = **+0.013**.
Routing cos(δ̄A,δ̄B): linear +1.00, mlp +1.00 (< 0.5 = genuine per-type routing, not one stronger global vector).
**S1.2 NOT green**: requires a conditional arm routing to different directions per type (cos < 0.5) AND Δ-R > global AND a legible type signal (probe > 75%). Not met — an arm that beats global without directional routing is just one stronger GLOBAL vector (prompt-distribution artifact); a low probe would mean the cue is washed out.
