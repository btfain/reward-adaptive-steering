# M1b — multi-turn capture: does observing u2 break the single-turn ceiling? — mt_wildchat_v1

145 contexts (all 9 moves scored), Prometheus+rubric reward, e5 router features, 16 seeds. Single-turn reference: capture ~18%, argmax dominated by one generic move.

## 1) Capture (bandit best-of-2)
- realized +0.630 · random +0.573 · **context-blind top-2 +0.639** · oracle +0.751 (reward = swing_pw.npz)
- headroom captured vs random = 31% [23%, 38%] (beating random only means AVOIDING bad moves — not routing).
- **routing gain OVER context-blind top-2 = -0.010 [-0.022, +0.002]** => NULL: per-context routing adds nothing over a fixed best pair — the value is move SELECTION (marginal quality), not context-adaptive routing.

## 2) Heterogeneity (argmax move per context)
- top move 'deepen' wins 26% of contexts; 9/9 moves win at least one context.
  null:23, tighten:15, deepen:37, fix_error:23, apply_revision:12, ask_clarify:13, simplify:3, refuse_safely:10, ground_facts:9
- => moves are CONTEXT-DEPENDENT (no single dominant move) — the routable structure the RM lacked.

## 3) Separation over NULL (no intervention)
- some move beats null in 84% of contexts; mean(best-null) +0.25, mean(best-2nd) +0.11 (score points).
  per-move mean lift vs null: deepen+0.06, ask_clarify+0.02, null+0.00, fix_error-0.00, ground_facts-0.00, apply_revision-0.03, refuse_safely-0.10, tighten-0.12, simplify-0.13

## 4) Length decoupling (the folded-in guard)
- (pairwise npz has no per-sample scores; length guard comes from the absolute run: -0.11).
