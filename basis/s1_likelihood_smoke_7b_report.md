# P1(i) LIKELIHOOD smoke — does the move explain the contrast? (reward-free, interpretability)

2199 candidates with both signals. Preference-shift kept 1359 (62%).

## Reward vs preference validation — do they agree?
- correlation(reward swing, preference shift) = **+0.099**
- both pass: 593 | reward-only: 280 | preference-only: 766 | neither: 560
- ⇒ reward∧preference agree on 52% of candidates; 280 raise reward WITHOUT explaining the contrast (works for another reason), 766 explain the contrast WITHOUT raising on-policy reward.

## Reading
- LOW correlation ⇒ reward and mechanism diverge — many moves raise reward for reasons OTHER than the guessed contrast (or vice versa); the two filters are genuinely different views, worth reporting.
