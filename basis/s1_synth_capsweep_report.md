# S1.1 — learned reward-driven steering (synthetic positive control)

Target e* over ('words', 'hedge_per100', 'questions_per100') = [-1.0, 1.0, 0.0] (z-scored). SmolLM2-1.7B, layer 16, mag cap 194 (=0.25·ref 778), rank 1 global.

## On-policy reward (held-out test prompts)
- base R:   **+0.072**
- steered R: **+0.549**
- **Δ-R = +0.477 [+0.215, +0.780]**  (95% boot CI; per-prompt paired; 21/32 prompts up)
- train-pool best-of-12 ceiling: +1.649

## Recovery — realized φ shift (steered − base), should align with e*
| feature | e* | base | steered | Δ | aligned |
|---|---|---|---|---|---|
| words | -1 | 78.08 | 77.78 | -0.30 | ✓ |
| hedge_per100 | +1 | 0.35 | 0.67 | +0.32 | ✓ |
| questions_per100 | +0 | 0.14 | 0.11 | -0.04 | ✓ |

## Cosine of learned direction to contrastive axes
- warm_neutral: -0.04
- challenge_accommodate: -0.03
- cautious_direct: -0.02
- inquire_proceed: -0.02
- elaborate_concise: +0.01
- formal_casual: +0.01
- hedge_assert: -0.01

## Reading
GREEN-for-S1.1: Δ-R > 0 on held-out prompts AND φ moves toward e* — the RWR machinery learns a reward-aligned steering direction from scratch. This is the machinery check before the real RM (S1.2).
