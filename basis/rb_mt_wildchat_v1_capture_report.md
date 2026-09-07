# M1b — multi-turn capture: does observing u2 break the single-turn ceiling? — mt_wildchat_v1

150 contexts (all 9 moves scored), Prometheus+rubric reward, e5 router features, 16 seeds. Single-turn reference: capture ~18%, argmax dominated by one generic move.

## 1) Capture (bandit best-of-2)
- realized +4.030 · random +4.031 · oracle +4.505 (reward = swing.npz)
- **headroom captured = -4% [-16%, 9%]** => not clearly above the single-turn ~18% — observing u2 did not unlock routing here.

## 2) Heterogeneity (argmax move per context)
- top move 'null' wins 28% of contexts; 9/9 moves win at least one context.
  null:42, tighten:31, deepen:18, fix_error:19, apply_revision:9, ask_clarify:5, simplify:5, refuse_safely:13, ground_facts:8
- => moves are CONTEXT-DEPENDENT (no single dominant move) — the routable structure the RM lacked.

## 3) Separation over NULL (no intervention)
- some move beats null in 72% of contexts; mean(best-null) +0.81, mean(best-2nd) +0.25 (score points).
  per-move mean lift vs null: fix_error+0.20, refuse_safely+0.06, ground_facts+0.03, ask_clarify+0.03, tighten+0.02, null+0.00, apply_revision-0.09, deepen-0.10, simplify-0.23

## 4) Length decoupling (the folded-in guard)
- corr(judge score, response length) = **-0.11** (low => our rubric is length-decoupled, unlike the style-biased RM).
