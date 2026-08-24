# Problem statement — interpretable proposal construction for inference-time reward optimization

*(Records/spine for Subproject 1. Companion to PLAN.md; formalizes what candidate generation (i),
submodular basis selection (ii), and router training + top-k selection (iii) jointly optimize, and
states when the method beats naive best-of-n.)*

## 1. Setup and target

- Frozen base policy `π₀(y|x)`, reward model `r(x,y)`, prompt law `x∼D`, inverse temperature `β`.
- Inference-time alignment target = the KL-regularized optimal policy
  **`π*(y|x) ∝ π₀(y|x)·exp(r(x,y)/β)`** — a reward-tilt of `π₀`.
- The **reward-optimal subregion of the KL-ball** for a prompt `x` is where `π*` concentrates mass
  and `π₀` does not. Every inference-time method (best-of-n, rejection sampling, guided decoding,
  SMC) is an approximate sampler of `π*`; they differ in how compute is spent to cover that subregion.

## 2. Moves as proposals

- An **action / move** `m` — a procedural system prompt, or an activation-steering vector `δ` —
  induces a shifted proposal `π_m(·|x)`: still near `π₀` (bounded KL for a mild move), but with mass
  moved toward some region.
- Candidate pool `X = {m₁,…,m_C}` produced by stage (i).
- Per-move value `v(x,m) = E_{y∼π_m}[r(x,y)]`; per-sample draw `r(x,y_m)`, `y_m∼π_m`.
- **Deployed best-of-set value** (generate one sample per move in `T`, keep reward-best):
  **`V(x,T) = E[ max_{m∈T} r(x,y_m) ]`** — captures both the mean *shift* and the within-move sampling
  *tail*. This is the quantity the bake-off measures.

## 3. The nested objective (stages i–iii as one optimization)

Test-time budget `k` generations/prompt; basis size `K ≥ k`:

```
maximize_{S ⊆ X, |S| ≤ K}   maximize_{ρ : ρ(x) ⊆ S, |ρ(x)| = k}   J(S,ρ) = E_{x∼D}[ V(x, ρ(x)) ]
```

- **(iii) Router `ρ`** (inner max, given `S`): `ρ*(x) = argmax_{T⊆S,|T|=k} V(x,T)`. This is a
  **ranking/coverage** objective — the router only needs the true-best move to land in its top-`k`,
  NOT to predict its value. This is the formal reason **top-k selection succeeds where top-1
  prediction is bounded** (B1: top-1 = single; bake-off: router top-2 > naive). At test time the
  reward-best of the `k` generations is kept (β→0 limit = argmax), so router error that keeps the
  winner in-set is harmless.
- **(ii) Basis selection** (outer max, given `X`):
  - *k = K regime* (generate all of `S`): `E_x[V(x,S)]` is monotone **submodular** in `S`
    (expected-max / weighted-coverage) ⇒ greedy is `(1−1/e)`-optimal. Stage (ii) is *derived*, not
    assumed.
  - *K > k regime* (router picks `k` of `K` per prompt — unlocked by top-k + a trained router):
    `S` is a **menu**; `F_k(S)=E_x[max_{|T|=k,T⊆S} V(x,T)]` has diminishing returns (approximately
    submodular), greedy still applies. Because test-time cost is `k`, not `|S|`, stages (i)/(ii) can
    afford a **large candidate pool `K`** and pay only at selection time. Marginal-gain estimates
    must be computed under a **generation-cost oracle** ⇒ cost-aware (lazy/CELF) submodular
    maximization with sample-based marginal estimation (secondary contribution).
- **(i) Candidate generation** maximizes attainable `F` over pools `X`: propose moves that shift mass
  onto reward-optimal subregions **poorly covered by the current basis** (diversity/complementarity-
  seeking, not average-value-seeking — marginals are context-dependent past move #1).

## 4. Baseline and efficiency claim

- **Naive best-of-N** is the special case `S = {no-shift}` sampled `k` times:
  `B(k) = E_x[ E[max of k iid π₀ draws] ]`. It is itself an approximation to `π*`
  (`KL(BoN‖π₀) ≈ log N − (N−1)/N`), which is why it is the honest bar.
- **Claim (reward-vs-cost Pareto):** `J(S*,ρ*)` at budget `k` `>` `B(k')` for `k' ≫ k`
  (bake-off: `k=2` router-moves ≈ `k'≈4` base samples; router-move > naive at every `k`).
- **Win condition (crisp, per-prompt testable):** move-selection beats BoN exactly when a prompt's
  reward-optimal subregion is **low-probability under `π₀` but reachable by a bounded-KL move** —
  a *distinct mode*, not a fat tail. If the high-reward region is already in `π₀`'s tail (fat-tailed
  `r`), BoN samples it cheaply and the gap closes; if it is a separated mode a mild move shifts onto,
  moves cover it in `O(1)` samples vs BoN's `O(1/π₀(region))`.

## 5. Unifying interpretation — interpretable reward-importance sampling

The routed basis is an **adaptive, interpretable mixture proposal** `q_x = π_{ρ(x)}` for importance-
sampling the tilted target `π*`. Naive BoN uses the base proposal `q = π₀`; we instead **learn**:
(i) a small set of human-readable proposal components (moves = interpretable modes of `π*`),
(ii) a submodular-minimal basis covering the reward-optimal subregions across `D`,
(iii) a per-prompt router selecting which components to sample; best-of-k realizes the selection.
**No generator training, no token-level machinery** — the niche vs reward-guided decoding
(Controlled Decoding / RAD / FUDGE / PPLM) and vs principled-but-opaque samplers (twisted SMC).

## 6. Baseline landscape (all on the reward-vs-cost / reward-vs-KL frontier)

| family | methods | role for us |
|---|---|---|
| sampling/selection (ours) | naive best-of-N; soft/weighted BoN; (quasi-)rejection sampling; adaptive/speculative BoN | **primary empirical baselines** (same-proposal controls) |
| basis-without-router (ours, ablation) | random-move / all-move best-of-k | **isolates the router's contribution** (≈ BoN ⇒ ranking is load-bearing) |
| reward-guided decoding | Controlled Decoding; RAD; FUDGE/GeDi; PPLM (decode-time steering) | cited stronger baseline; positioned against on interpretable / no-token-machinery axis |
| principled importance sampling | twisted SMC; QRS | theoretical target we approximate cheaply/interpretably |
| adjacent (out of scope) | self-refine/self-correct; self-consistency / ToT | different axis (revision / reasoning), excluded |

Report axes: **reward vs #generations** (primary) and **reward vs KL-from-π₀** (secondary); tag prompts
by the mode-vs-tail win-condition to test *where* the method helps.

## 7. What is established vs open (as of 2026-08-24)

- **Established — prediction is bounded (B1):** fine-tune = frozen = single (paired +0.002), even at
  1500 prompts / 124.7M params. Conditioning-by-prediction from the prompt does not exceed the
  unconditional lever.
- **Established — selection works (bake-off):** router-move best-of-2 `+0.992` > naive best-of-2
  `+0.629` (paired `+0.350 [+0.160,+0.535]`, ~2× compute-efficient); random-move-2 ≈ naive ⇒ the
  **router's ranking**, not the basis alone, drives the win.
- **Open:** (1) how much of the router→oracle gap (`+0.99` vs `+1.53` at k=2) a **stronger ranker**
  closes; (2) whether **steering directions** admit a covering basis (steering-by-selection) or the
  high-rank coverage wall kills it even for selection; (3) the mode-vs-tail win-condition, per-prompt.
