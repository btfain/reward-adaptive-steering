# Stage B0 report — synthetic positive control (manifestation dial)

630 prompts (6 types x explicit/situational/none), 25 actions, noiseless test-set values. oracle_type = recoverable ceiling (true type known); oracle_prompt additionally includes generation luck; skyline = supervised-probe posterior plug-in (feature-set ceiling, diagnostic only). Probe type-accuracy (test): explicit 0.80, situational 0.77, none 0.09 (chance 0.17; none ~ chance is the designed MI=0 floor.)

| beta | d | oracle_prompt | oracle_type | skyline | none | fixed | conditional |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.0 | +1.49 | +1.42 | +1.42 | +0.01 | +1.42 | +0.98±0.01 |
| 0.3 | 0.5 | +1.15 | +0.94 | +0.80 | +0.03 | +0.84 | +0.58±0.04 |
| 0.3 | 1.0 | +0.73 | +0.48 | +0.21 | +0.05 | +0.05±0.07 | +0.09±0.02 |
| 1.0 | 0.0 | +4.97 | +4.75 | +4.75 | +0.03 | +4.75 | +3.35±0.04 |
| 1.0 | 0.5 | +3.84 | +3.15 | +2.68 | +0.10 | +2.80 | +1.93±0.10 |
| 1.0 | 1.0 | +2.44 | +1.61 | +0.69 | +0.15 | +0.27±0.06 | +0.49±0.10 |
| 3.0 | 0.0 | +14.91 | +14.24 | +14.24 | +0.08 | +14.24 | +9.81±0.04 |
| 3.0 | 0.5 | +11.53 | +9.45 | +8.03 | +0.29 | +8.41 | +6.04±0.31 |
| 3.0 | 1.0 | +7.31 | +4.83 | +2.07 | +0.45 | +0.73±0.01 | +1.54±0.33 |

## Money table: conditional minus fixed (test value) vs type-dependence d

| beta | d=0.0 | d=0.5 | d=1.0 |
|---|---|---|---|
| 0.3 | -0.44 | -0.26 | +0.04 |
| 1.0 | -1.40 | -0.87 | +0.22 |
| 3.0 | -4.43 | -2.37 | +0.82 |

## Value capture vs the type-oracle, by manifestation ((arm - none)/(oracle_type - none); '—' where ceiling ~ floor)

| beta | d | fixed exp/sit/none | conditional exp/sit/none | skyline exp/sit/none |
|---|---|---|---|---|
| 0.3 | 0.0 | +1.00 / +1.00 / +1.00 | +0.76 / +0.52 / +0.77 | +1.00 / +1.00 / +1.00 |
| 0.3 | 0.5 | +0.94 / +0.83 / +0.88 | +0.67 / +0.53 / +0.59 | +0.90 / +0.90 / +0.72 |
| 0.3 | 1.0 | — / -0.04 / — | — / +0.36 / — | — / +0.67 / — |
| 1.0 | 0.0 | +1.00 / +1.00 / +1.00 | +0.76 / +0.54 / +0.80 | +1.00 / +1.00 / +1.00 |
| 1.0 | 0.5 | +0.94 / +0.83 / +0.88 | +0.66 / +0.57 / +0.57 | +0.90 / +0.90 / +0.72 |
| 1.0 | 1.0 | +0.14 / +0.02 / +0.11 | +0.58 / +0.54 / -0.35 | +0.61 / +0.67 / -0.14 |
| 3.0 | 0.0 | +1.00 / +1.00 / +1.00 | +0.78 / +0.53 / +0.74 | +1.00 / +1.00 / +1.00 |
| 3.0 | 0.5 | +0.94 / +0.83 / +0.88 | +0.70 / +0.55 / +0.63 | +0.90 / +0.90 / +0.72 |
| 3.0 | 1.0 | +0.09 / +0.01 / +0.10 | +0.55 / +0.62 / -0.37 | +0.61 / +0.67 / -0.14 |

Secondary: action axis+sign match with the type-optimal action at beta=3.0, d=1: explicit +0.27±0.11, situational +0.23±0.05, none +0.13±0.01. Low match with high value capture = the policy hedges under type uncertainty (correct behavior).

Type-optimal actions at d=1: {np.str_('assertive'): 'hedge_assert-0.2', np.str_('concise'): 'inquire_proceed+0.2', np.str_('elaborate'): 'hedge_assert+0.2', np.str_('hedged'): 'hedge_assert+0.2', np.str_('inquiring'): 'inquire_proceed+0.2', np.str_('proceeding'): 'formal_casual+0.2'}
Note: cross-axis effects dominate the words feature (the 96-token cap right-censors elaborate steering; inquire+0.2 is the strongest word-count reducer), so type-optimal actions need not sit on the 'matching' axis. The learner's job is the reward-optimal mapping, whatever its geometry.

## Greedy action distribution by cell (beta=3.0, d=1; top picks)

| type | explicit | situational | none |
|---|---|---|---|
| assertive | elaborate_concise-0.2 (58%), hedge_assert+0.2 (15%) | cautious_direct-0.2 (33%), warm_neutral-0.2 (30%) | inquire_proceed+0.2 (24%), formal_casual+0.2 (12%) |
| concise | elaborate_concise-0.2 (67%), hedge_assert-0.2 (15%) | elaborate_concise-0.2 (88%), inquire_proceed+0.2 (6%) | inquire_proceed+0.2 (36%), elaborate_concise-0.2 (24%) |
| elaborate | cautious_direct+0.2 (48%), hedge_assert+0.2 (33%) | cautious_direct+0.2 (39%), hedge_assert+0.2 (24%) | inquire_proceed+0.2 (33%), elaborate_concise+0.1 (21%) |
| hedged | hedge_assert+0.2 (73%), cautious_direct+0.2 (21%) | cautious_direct+0.2 (45%), cautious_direct-0.2 (15%) | inquire_proceed+0.2 (33%), elaborate_concise-0.2 (21%) |
| inquiring | warm_neutral+0.1 (30%), elaborate_concise+0.1 (30%) | inquire_proceed+0.2 (52%), inquire_proceed+0.1 (12%) | elaborate_concise-0.2 (24%), inquire_proceed+0.2 (24%) |
| proceeding | elaborate_concise-0.2 (36%), hedge_assert+0.2 (12%) | cautious_direct+0.2 (33%), hedge_assert+0.2 (24%) | inquire_proceed+0.2 (45%), elaborate_concise-0.2 (9%) |

Learner: GRPO-style (K=8 group-relative), linear head on layer-16 hidden states, weight decay 0.001, early stopping on a stratified val split (0.2 of train). Without both regularizers the 2048-dim head memorizes per-prompt generation luck (train +4.7 / test +1.4 at beta=3, d=1) — a direct Stage C lesson.

wall time 41s (local, cpu)
