# P1(i) — candidate pool evaluation (auto vs curated vs combined) — candpool_7b

Greedy submodular value f(S)=Σ_x max(0,max swing) on the SAME 96 prompts; only the candidate pool varies. Pool: 200 auto + 20 curated = 220.

| K | auto-only | curated-only | combined |
|---|---|---|---|
| 1 | +0.970 | +0.828 | +0.970 |
| 2 | +1.329 | +1.148 | +1.329 |
| 3 | +1.526 | +1.283 | +1.526 |
| 4 | +1.651 | +1.370 | +1.651 |
| 5 | +1.757 | +1.444 | +1.757 |
| 6 | +1.835 | +1.471 | +1.835 |
| 7 | +1.904 | +1.497 | +1.904 |
| 8 | +1.962 | +1.519 | +1.967 |
| 9 | +2.018 | +1.537 | +2.019 |
| 10 | +2.058 | +1.554 | +2.065 |
| 11 | +2.097 | +1.569 | +2.098 |
| 12 | +2.129 | +1.581 | +2.128 |
| 13 | +2.155 | +1.593 | +2.157 |
| 14 | +2.181 | +1.605 | +2.184 |
| 15 | +2.205 | +1.610 | +2.210 |
| 16 | +2.229 | +1.614 | +2.232 |

## Combined basis composition (top-16): 15 auto / 1 curated
  1:auto  2:auto  3:auto  4:auto  5:auto  6:auto  7:auto  8:curated  9:auto  10:auto  11:auto  12:auto  13:auto  14:auto  15:auto  16:auto

## Reading
- auto-only vs curated-only at K=8: +1.962 vs +1.519 ⇒ auto pool ≥ curated (automation matches/beats hand-crafting).
- combined − max(auto,curated) at K=8: +0.006 ⇒ does mixing pools add coverage (complementarity)?
- combined basis draws 15/16 from auto ⇒ the auto pool contributes substantially.
