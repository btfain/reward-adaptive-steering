# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→hedge− (z-scored φ). SmolLM2-1.7B, steer L16, read L16, rank 2, mag cap 142 (a_max 101). n=12 pool. Explicit type cue. Δ-R = on-policy steered − base, held-out.

**Type-separability probe** (linear h(x)→type): train 100%, held-out **73%** — near 100% ⇒ routing signal present (failure would be optimization/reward); ~50% ⇒ cue washed out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.029 [-0.119, +0.161] | -0.042 | +0.100 |
| linear | -0.068 [-0.258, +0.106] | +0.271 | -0.406 |
| mlp | -0.027 [-0.129, +0.064] | -0.049 | -0.005 |

## Routing — mean coefficient a per type (gap = ||a|A − a|B|| / a_max)
- **global**: a|A = [26.0, -12.29], a|B = [26.0, -12.29] → gap **0.00**
- **linear**: a|A = [-100.54, 100.54], a|B = [-100.54, 100.54] → gap **0.00**
- **mlp**: a|A = [100.54, 100.54], a|B = [100.54, 100.54] → gap **0.00**

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +1.63, Δhedge -0.03, Δquestions -0.08
- **global** type B: Δwords +0.56, Δhedge -0.08, Δquestions +0.05
- **linear** type A: Δwords +2.26, Δhedge +0.22, Δquestions -0.04
- **linear** type B: Δwords +0.72, Δhedge +0.32, Δquestions +0.05
- **mlp** type A: Δwords +6.12, Δhedge -0.04, Δquestions -0.09
- **mlp** type B: Δwords +3.24, Δhedge +0.00, Δquestions -0.07

## Reading
Conditioning value = best conditional Δ-R (-0.027, mlp) − global Δ-R (+0.029) = **-0.056**.
Routing gaps: linear 0.00, mlp 0.00 (need ≥ 0.3 to count as conditioning, not a stronger global vector).
**S1.2 NOT green**: requires a conditional arm with routing gap ≥ 0.3 AND Δ-R > global AND a legible type signal (probe > 75%). Not met — arms that beat global without a routing gap are just stronger GLOBAL vectors (prompt-distribution artifact), not conditioning; a low probe would mean the cue is washed out.
