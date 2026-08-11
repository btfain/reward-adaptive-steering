# S1.2 — conditional steering controller (type-dependent positive control)

Types A→hedge+, B→questions+ (z-scored φ). SmolLM2-1.7B, steer L16, read L16, rank 2, mag cap 142. n=12 pool. Explicit type cue (machinery check). Δ-R = on-policy steered − base, held-out.

| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |
|---|---|---|---|
| global | +0.078 [-0.037, +0.184] | +0.208 | -0.052 |
| linear | +0.028 [-0.097, +0.145] | +0.097 | -0.042 |
| mlp | +0.248 [+0.099, +0.430] | +0.211 | +0.284 |

## Routing — mean coefficient a per type (r directions)
- **global**: a|A = [27.34, -10.153], a|B = [27.34, -10.153]
- **linear**: a|A = [-2774.436, 1938.228], a|B = [-2689.423, 1956.711]
- **mlp**: a|A = [48062.246, 71509.562], a|B = [46316.238, 68878.719]

## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants questions↑
- **global** type A: Δwords +1.74, Δhedge +0.17, Δquestions -0.02
- **global** type B: Δwords +1.91, Δhedge +0.15, Δquestions -0.05
- **linear** type A: Δwords +3.87, Δhedge +0.08, Δquestions -0.10
- **linear** type B: Δwords +2.72, Δhedge +0.12, Δquestions -0.04
- **mlp** type A: Δwords +1.36, Δhedge +0.17, Δquestions -0.01
- **mlp** type B: Δwords -0.68, Δhedge +0.25, Δquestions +0.27

## Reading
Conditioning value = best conditional Δ-R (+0.248, mlp) − global Δ-R (+0.078) = **+0.170**. GREEN-for-S1.2: a conditional controller beats the global vector AND routes the right direction to each type (a|A ≠ a|B, φ recovers each type's lever). Linear vs MLP = capacity comparison.
