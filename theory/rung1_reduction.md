# Rung 1 — Reduction: single-turn reward optimization is a greedy policy

> **Purpose.** This note is the theoretical spine of the multi-turn study. It is a
> *reduction*, not an experiment: it states formally that single-turn LLM reward
> optimization is a one-step-greedy policy in the underlying sequential decision
> problem, and that its regret is the value of non-myopic action. The gap it
> identifies is standard RL — greedy maximization of a per-step reward is not
> return-optimal — and it holds in the **fully observed** MDP; partial
> observability is not required. The contributions are (i) the explicit mapping
> onto how LLMs are trained, (ii) the corollary that **longer training context
> cannot close the gap**, and (iii) the observation that Stage A2 already exhibits
> a concrete instance with strictly positive regret. The experimental teeth — "and
> it materially changes LLM behavior" — come from the Rung 2 verifiable
> environment, not from a synthetic RL run here.

---

## 1. Setup: the interaction as an MDP (fully observed)

Model a multi-turn interaction as an MDP `M = (S, A, P, r, γ)`. Nothing here is
partially observed; that specialization is deferred to §4.

- **State.** `s_t` is everything the agent conditions on — the full conversation
  history `h_t = (o₁, a₁, …, o_t)`. Longer training context = a larger `s_t`.
- **Action.** `a_t ∈ A` is the agent's turn (a response). In our method the action
  is *frozen base generation modulated by a low-dimensional control* (steer or
  prompt-selection); for the reduction that decomposition is irrelevant and `a_t`
  is simply the response.
- **Transition.** `s_{t+1} ∼ P(· | s_t, a_t)` — the environment (user + world)
  returns the next turn. The agent's action *shapes the distribution of future
  states*.
- **Reward.** The learned reward model gives a **per-step** score `r(s_t, a_t)`: it
  rates a single response, with no dependence on the continuation. The *task* value
  is the return `Σ_t γ^t r(s_t, a_t)`.

---

## 2. The single-turn objective **is** the greedy policy

Single-turn RLHF / reward training optimizes, for the learned reward `r` and a fixed
distribution `D` of contexts,

```
J_bandit(π)  =  E_{s ~ D}  E_{a ~ π(·|s)} [ r(s, a) ].
```

This is a **contextual bandit**: each `(s, a)` is scored independently by its
immediate reward, summed over contexts drawn from a *fixed, exogenous* `D`, with no
coupling through transitions. Its optimum is the myopic policy

```
π_g(s)  =  argmax_a  r(s, a).            (greedy / single-turn-optimal)
```

Contrast the sequential objective the interaction actually poses,

```
J_seq(π)  =  E [ Σ_t γ^t r(s_t, a_t) ] ,   s_{t+1} ~ P(·|s_t,a_t).
```

Two differences, both essential:

1. **State distribution.** In `J_seq` the distribution of `s_t` is *induced by π
   itself* (on-policy occupancy `d^π`); `J_bandit` freezes it to an exogenous `D`
   (the off-policy training set).
2. **Credit coupling.** In `J_seq` the reward at `t` depends on actions at `t' < t`
   through `P`; `J_bandit` severs that coupling — an action's effect on future
   states is never in the objective.

> **Reduction (statement).** Single-turn reward optimization = maximizing
> `J_bandit`, whose optimum `π_g` is the one-step-greedy policy of `M`: at every
> state it maximizes immediate reward and ignores continuation value.

---

## 3. Regret of the greedy policy = value of non-myopic action

Let `V*` be the optimal value of `M` and `V^{π_g}` the greedy policy's value. Greedy
maximizes the immediate reward `r(s,a)` in place of the state-action value

```
Q*(s,a)  =  r(s,a)  +  γ · E_{s' ~ P(·|s,a)} [ V*(s') ].
```

The term it drops,

```
Q*(s,a) − r(s,a)  =  γ · E_{s'}[ V*(s') ]     (the continuation value),
```

is precisely the downstream consequence of `a`. Hence:

- **Greedy is optimal iff the orderings agree:** `argmax_a r(s,a) = argmax_a Q*(s,a)`
  at every reachable `s`. Regret is nonzero exactly when some action's rank *flips*
  between `r` and `Q*` — when an action's continuation value outweighs its
  immediate-reward deficit. (Sufficient condition for *no* gap: `r` is `Q*`, or a
  potential-based shaping of it. Generic per-step rewards are neither.)
- By the performance-difference lemma,
  `V*(s₀) − V^{π_g}(s₀) = E_{s ~ d^{π*}}[ A^{π_g}(s, π*(s)) ] ≥ 0`,
  strictly positive whenever such a rank flip occurs on the optimal occupancy.

This is the essential point, and it is **fully observed**: the greedy use of a
per-step reward to optimize a trajectory return is myopic on its face. Everything
below is either a corollary or a way to make the gap concrete and non-vacuous.

---

## 4. A concrete, conversation-salient instance (value of information)

The rank-flip need not be exotic. The minimal fully-observed example is delayed
reward: an action with low `r` that transitions to a high-`V*` state (e.g. laying
groundwork, decomposing, verifying). The instance most relevant to conversation —
and the one Stage A2 instantiates — is **information gathering**, which *also*
introduces partial observability, though the gap does not depend on it.

Let the environment carry a latent `z` (the user's intent / acceptance criterion),
and let the agent hold a belief `b` over `z`. Two action types:

- **Ask** `a_ask`: a clarifying turn. Immediate reward `r(a_ask) = −c`, `c ≥ 0`; it
  returns an observation that sharpens the belief to the posterior `P(z | o)`.
- **Commit** `a_commit(g)`: a final answer tailored to a guess `g`. Terminal reward
  `R = 1` iff `g = z`, else `0`.

Committing now earns the modal-guess success `max_g b(g)`. Asking first, then
committing under the sharpened belief, earns `−c + γ · E_o[ max_g P(z=g | o) ]`. The
**value of information** is

```
VoI(b)  =  [ −c + γ · E_o max_g P(z=g | o) ]  −  max_g b(g).
```

- **Greedy** compares *immediate* rewards: `r(a_ask) = −c < 0 ≤ r(a_commit)`, so
  **greedy never asks** whenever committing has any nonnegative immediate reward.
- **Optimal** asks iff `VoI(b) > 0`.

> **Proposition (informal).** When the belief is diffuse enough that
> `γ·(expected posterior modal mass) − (prior modal mass) > c`, the greedy policy is
> strictly sub-optimal, regret `≥ VoI(b)`.
> *Proof sketch.* Greedy commits, earning `max_g b(g)`. "Ask then commit under the
> posterior" is feasible and earns `−c + γ E_o max_g P(z|o)`; the difference is
> `VoI(b)`, and `V* ≥` the feasible policy's value. ∎

This case is convenient because the flip is *guaranteed* (asking is always immediate-
negative) and *interpretable* — but to repeat: the §3 gap is the general fact, and
this is one illustration of it.

---

## 5. Corollary: longer context cannot close the gap

The formal answer to *"but we already train on long contexts."*

Enriching the context (larger `s_t`) changes only the **conditioning** of the greedy
policy. It does **not** change the **objective**: `π_g = argmax_a r(·)` still ignores
how `a_t` reshapes the distribution of future states. In the VoI instance this is
stark — a longer context can improve the prior `b(z|s)` but never makes `π_g` pay `c`
to ask — but the point is general: context enriches the *representation of the
present state*, while planning accounts for the *consequences of acting*. They
address different things:

| | resolves… | mechanism |
|---|---|---|
| **Long context** | uncertainty already implicit in the state | richer conditioning of a greedy `argmax` |
| **Multi-turn planning** | the *consequences of the action on future states* | accounting for continuation value `γE[V*(s')]` |

No amount of conditioning turns a per-step `argmax_a r` into `argmax_a Q*`. Longer
context and multi-turn planning are not substitutes; the former cannot supply what
the latter provides.

---

## 6. Why the gap is non-vacuous for real systems (A2 as the hinge)

The VoI Proposition's hypothesis has two empirical parts, and **Stage A2 already
verified both** for an actual base+RM pair:

1. **`c > 0`.** Reward models *penalize* clarifying questions. A2 measured
   `inquire_proceed` as single-turn-negative under **both** RMs (e.g.
   `m2:inquire_proceed+0.2` at ΔRM ≈ −4.1/−3.6; the whole `inquire+` region is the
   most-penalized in the sweep). Asking has strictly negative immediate reward.
2. **Diffuse belief exists.** UltraFeedback contains underspecified prompts whose
   acceptance criterion is not recoverable from the prompt — `max_g b(g)` well below
   1.

So the reduction upgrades A2's *null* into a provable *positive*: for this base+RM
pair there is a regime of strictly positive regret. A2 is not a failed search — it
is the measurement that instantiates the gap on a system people care about. The one
behavior single-turn training most reliably suppresses (asking) is exactly the one
with positive value of information.

---

## 7. Bridge to Rung 2: the optimal policy is *simple*

In the VoI instance the optimal policy is a **threshold on the belief**:

```
ask   if  VoI(b) > 0   (belief ambiguity exceeds a threshold),
commit to argmax_g b(g)   otherwise.
```

The *gap* can be large while the *planning needed to close it is shallow* — a
low-dimensional decision (estimate ambiguity → ask or commit), not deep value
iteration. This is the formal seed of the study's thesis:

> **Interpretable, simple planning is sufficient to close most of the gap.**

Rung 2 tests this — first with the action space handed to us (planning-move types =
the rubric's dimensions), then with the action space learned from reward. The reward
there is a **deterministic rubric check**, not a proxy.

---

## 8. Assumptions and boundaries

- **Fully observed suffices.** The gap is the myopic use of a per-step reward
  (§3); partial observability and the latent `z` are one *instance* (§4), not a
  requirement.
- **The single-turn reward `r` is immediate** — it scores one response with no
  dependence on the continuation. This is how deployed RMs work. *If* an RM were
  trained to predict long-horizon return, the premise `r ≈ immediate` weakens and
  the reduction's teeth dull — the gap is a property of the *training objective*,
  not of neural nets.
- **The interaction has a task value** (a return worth optimizing). Fine for
  clarification-gated tasks with a checkable outcome; weaker for open-ended chat
  with no defined terminal.
- **The greedy-vs-return gap is standard RL;** the contribution is the mapping to
  LLM training, the long-context corollary, and the A2 instantiation — not the
  existence of the gap.
```
