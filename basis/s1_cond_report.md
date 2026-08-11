# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→hedge− (z-scored φ). SmolLM2-1.7B, steer L16, read L16, rank 2, magnitude FIXED at cap 107 (direction-only routing; coeffs a∈[−1,1]). n=12 pool. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): train 100%, held-out **73%** — near 100% ⇒ routing signal present (failure would be optimization/reward); ~50% ⇒ cue washed out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | -0.048 [-0.216, +0.112] | +0.095 | -0.191 |
| linear | +0.054 [-0.124, +0.212] | +0.327 | -0.220 |
| mlp | -0.010 [-0.125, +0.103] | -0.050 | +0.030 |

## Routing — mean coefficient a∈[−1,1] per type (gap = ||a|A − a|B||)
- **global**: a|A = [0.44, -0.89], a|B = [0.44, -0.89] → gap **0.00**
- **linear**: a|A = [-1.0, -1.0], a|B = [-1.0, -1.0] → gap **0.00**
- **mlp**: a|A = [-1.0, -1.0], a|B = [-1.0, -1.0] → gap **0.00**

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +2.54, Δhedge +0.08, Δquestions -0.10
- **global** type B: Δwords +2.55, Δhedge +0.15, Δquestions +0.08
- **linear** type A: Δwords +2.22, Δhedge +0.26, Δquestions -0.02
- **linear** type B: Δwords +0.75, Δhedge +0.18, Δquestions +0.14
- **mlp** type A: Δwords +3.58, Δhedge -0.04, Δquestions -0.05
- **mlp** type B: Δwords +2.40, Δhedge -0.02, Δquestions -0.02

## Reading
Conditioning value = best conditional Δ-R (+0.054, linear) − global Δ-R (-0.048) = **+0.102**.
Routing gaps: linear 0.00, mlp 0.00 (need ≥ 0.3 to count as conditioning, not a stronger global vector).
**S1.2 NOT green**: requires a conditional arm with routing gap ≥ 0.3 AND Δ-R > global AND a legible type signal (probe > 75%). Not met — arms that beat global without a routing gap are just stronger GLOBAL vectors (prompt-distribution artifact), not conditioning; a low probe would mean the cue is washed out.
