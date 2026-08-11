# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→questions+ (z-scored φ). SmolLM2-1.7B, steer L16, read L16, rank 2, mag cap 142. n=12 pool. Explicit type cue (machinery check). Δ-R = on-policy steered − base, held-out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.072 [-0.021, +0.159] | +0.026 | +0.118 |
| linear | +0.295 [+0.136, +0.470] | +0.497 | +0.093 |
| mlp | -0.046 [-0.150, +0.048] | -0.073 | -0.019 |

## Routing — mean coefficient a per type (r directions)
- **global**: a|A = [25.944, -12.097], a|B = [25.944, -12.097]
- **linear**: a|A = [-100.537, 100.536], a|B = [-100.536, 100.536]
- **mlp**: a|A = [100.537, 100.537], a|B = [100.537, 100.537]

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +0.78, Δhedge +0.02, Δquestions +0.03
- **global** type B: Δwords +0.61, Δhedge +0.11, Δquestions +0.11
- **linear** type A: Δwords +2.32, Δhedge +0.40, Δquestions +0.02
- **linear** type B: Δwords +1.81, Δhedge +0.22, Δquestions +0.09
- **mlp** type A: Δwords +4.74, Δhedge -0.06, Δquestions -0.10
- **mlp** type B: Δwords +2.49, Δhedge -0.01, Δquestions -0.02

## Reading
Conditioning value = best conditional Δ-R (+0.295, linear) − global Δ-R (+0.072) = **+0.223**. GREEN-for-S1.2: a conditional controller beats the global vector AND routes the right direction to each type (a|A ≠ a|B, φ recovers each type's lever). Linear vs MLP = capacity comparison.
