# Multi-turn control — move basis + rubric (v1, M1 prototype)

Approved 2026-09-06. Machine-readable source of truth: `configs/moves_multiturn_v1.yaml`.

## Setting
Given a logged chat prefix `(u1, a1, u2)`, a cheap controller/classifier picks **one** move; the move is
injected as a **system message** before the base model regenerates the final assistant turn `a2`. The new
information is the user's reaction `u2` — this is prospective **control**, not self-refine. Reward =
Prometheus-2-7B applying the context-appropriateness rubric below. Moves are fixed, offline-vetted **text**
(never generated at inference) → safe a priori + interpretable; the controller emits only a discrete index.

## Design principles
- **Each move = (recognizable trigger in `u2`) → (detailed, vetted action)** — the "bank of generalized
  safety classifiers" made concrete ("harmful → refuse" is move 7).
- **Triggers are readable from `u2`** ("too long", "that's wrong", "I don't get it", turns unsafe) → the
  right move is genuinely a function of observed state, the thing missing single-turn. This is the bet M1
  tests (does capture beat the single-turn ~18%?).
- **Length/style-neutral rubric** → moves separate by *context-fit*, not verbosity: TIGHTEN wins when
  brevity is asked and *loses* to DEEPEN when depth is asked → no single dominant move (the failure we
  kept hitting on the style-biased RM).

## Rubric (Prometheus-style, FLASK-grounded, length-neutral)
See `configs/moves_multiturn_v1.yaml` `rubric:` for the exact text. Criterion = context-appropriateness of
`a2`; correctness + harmlessness are prerequisites (cap at 2); conciseness is rewarded, length is not.
FLASK skills folded in: comprehension, completeness, conciseness, correctness/factuality, harmlessness.

## Moves (9) → rubric dimension
| # | name | trigger in `u2` | rubric dim |
|---|---|---|---|
| 0 | null | baseline, no intervention | — |
| 1 | tighten | too long / "just the answer" / shorter | conciseness |
| 2 | deepen | "why" / "explain more" / "walk me through" | completeness / comprehension |
| 3 | fix_error | flags a mistake ("that's wrong", "actually X") | correctness + responsiveness |
| 4 | apply_revision | specific change ("make it formal", "in Python") | responsiveness / conciseness |
| 5 | ask_clarify | genuinely ambiguous; needs unknown details | responsiveness (clarification) |
| 6 | simplify | confused / "I don't get it" / too technical | comprehension / readability |
| 7 | refuse_safely | `u2` turns harmful/disallowed | harmlessness (rejection) |
| 8 | ground_facts | questions a factual claim / claim is checkable | factuality / correctness |

## Status / next
- Basis size 9 for the prototype; **expand later during move-DISCOVERY** (learn moves from real user
  mid-chat interventions + safety classifiers).
- Feeds **M1a** (judge/reward validation: Prometheus + this rubric, length-decoupling + published-agreement
  gate) then **M1b** (WildChat swing matrix + capture/separation).
