# Stage 1 — theory (T1–T4). Companion to PROBLEM.md (spine) and STAGE1_COMPLETION.md (plan).

## Notation
- Candidate moves `V`, `|V| = C`. Prompts `x ∼ D`; empirical objective uses `N` prompts.
- Swing `w(x,p) ∈ ℝ`. Per-prompt contribution of a set `S`:
  `h_x(S) = max(0, max_{p∈S} w(x,p))` (with `h_x(∅)=0`).
- **Objective** `f(S) = E_{x∼D}[h_x(S)]`, and its empirical version on a prompt set `B`:
  `f̂_B(S) = (1/|B|) Σ_{x∈B} h_x(S)`.
- `R := ess sup_{x,p} max(0, w(x,p))` — the per-prompt contribution is bounded in `[0, R]`.
- `OPT := max_{|S|≤K} f(S)`.

**Fact (submodularity).** For each `x`, `h_x(S)=max({0}∪{w(x,p):p∈S})` is a monotone submodular set
function (max-of-coverage; the marginal of adding `p` is `max(0, w(x,p) − current max)`, which is
non-increasing in `S`). Hence `f` and every `f̂_B` are monotone submodular ⇒ greedy is `(1−1/e)`-optimal.

Querying one cell `(x,p)` (running one generation to get `w(x,p)`) is the unit of cost. Full greedy needs
all `C·N` cells. **T1 shows prompt subsampling recovers a near-greedy basis at `C·s` cells, `s ≪ N`.**

---

## T1 — Subsample-and-select (prompt subsampling)

**Theorem T1.** Fix `ε_conf = δ`, `η > 0`. Draw a prompt sample `B` of size
`s ≥ (R²/(2η²))·(K·ln C + ln(2/δ))` i.i.d. from `D`, and let `Ŝ` be the greedy solution of `f̂_B` under the
cardinality-`K` constraint. Then with probability at least `1−δ` (over `B`),

  `f(Ŝ) ≥ (1 − 1/e)·OPT − 2η`,

using at most `C·s` cell-queries — a factor `N/s` fewer generations than full greedy, **independent of N**.

**Proof.**
1. *(Uniform convergence / SAA.)* For any fixed `S`, `h_x(S)∈[0,R]` and `f̂_B(S)` is an average of `s`
   i.i.d. terms, so Hoeffding gives `Pr[|f̂_B(S)−f(S)| > η] ≤ 2 exp(−2sη²/R²)`. There are at most
   `Σ_{k≤K} \binom{C}{k} ≤ C^K` candidate sets of size `≤ K`. Union bound with
   `2 exp(−2sη²/R²) ≤ δ/C^K`, i.e. `s ≥ (R²/2η²)(K ln C + ln(2/δ))`, yields
   `sup_{|S|≤K} |f̂_B(S) − f(S)| ≤ η` w.p. `≥ 1−δ`. Call this event `𝓖`.
2. *(Greedy on the surrogate.)* On `𝓖`, let `S* = argmax_{|S|≤K} f`. Greedy on `f̂_B` returns `Ŝ` with
   `f̂_B(Ŝ) ≥ (1−1/e)·max_{|S|≤K} f̂_B(S) ≥ (1−1/e)·f̂_B(S*) ≥ (1−1/e)(f(S*) − η)`.
3. *(Transfer back.)* Again on `𝓖`, `f(Ŝ) ≥ f̂_B(Ŝ) − η ≥ (1−1/e)(OPT − η) − η ≥ (1−1/e)·OPT − 2η`. ∎

**Cost.** Greedy on `f̂_B` evaluates every candidate's marginal on the `s` sampled prompts; cells are
cached, so total unique queries `= C·s`. Saving factor `N/s`.

---

## T1′ — candidate axis (stochastic greedy), and an honest caveat
Composing **stochastic greedy** (Mirzasoleiman et al. 2015: evaluate a random subset of
`r = ⌈(C/K)ln(1/ε)⌉` candidates per step) on top of T1 gives, in expectation over the subsets,
`E[f(Ŝ)] ≥ (1 − 1/e − ε)·OPT − 2η`, reducing marginal *function-evaluations* from `KC` to `≈ C·ln(1/ε)`.

**Caveat (measured, not just stated).** In our regime this does **not** reduce *generations*: with cells
cached, the unique candidates ever queried under stochastic greedy is
`≈ C(1 − e^{−rK/C}) = C(1−ε) ≈ C`, so cell cost stays `≈ C·s`. Cutting the candidate axis in *generations*
requires eliminating candidates *before* querying them on any prompt — an adaptive best-arm-identification
problem. We tried exact Hoeffding elimination and it saved nothing: candidate marginals are close (gaps
`≪ R`), so distinguishing the argmax needs `≈ N` samples. **So the generation saving is the prompt axis
(T1); the candidate axis is an open problem** (variance-adaptive Bernstein / ε-approximate best-arm under
a cost oracle) — a clean secondary contribution if solved, honestly reported as open if not.

---

## Empirical support (offline, `src/cost_submod.py`)
Cost-aware greedy vs full greedy on the real `large_7b` matrix (C=20) and synthetic (C=100/200):

| pool | s | value (% of full-greedy) | cells (% of C·N) |
|---|---|---|---|
| C=200 | 32 | 97% | 11% |
| C=200 | 64 | 98% | 21% |
| C=200 | 128 | 99% | 43% |

Consistent with T1 (value `→` full-greedy as `s` grows; cells `= C·s = (s/N)·CN`). **Practice beats the
worst-case bound:** T1 guarantees only `(1−1/e)OPT − 2η ≈ 0.63·OPT`, but greedy is near-optimal on benign
instances, so we observe ~95–99% of full-greedy value at `s = 32–64 ≪ N`. The `s` T1 prescribes
(`∝ R²/η²·K ln C`) is conservative (adversarial-Hoeffding); real marginals need far fewer samples.

---

## T2–T4 (targets; developed with their experiments)
- **T2 (ranking→reward).** Bound realized best-of-`k` reward by the router's top-`k` recall
  `Pr[argmax move ∈ top-k]`; ties ranking quality (iii) to the banner metric.
- **T3 (when selection beats BoN).** Under a mixture-of-modes model of the reward-tilted target
  `π*∝π₀e^{r/β}`, characterize when routed selection Pareto-dominates best-of-n on reward-vs-cost
  (formalizes the "bounded-KL mode vs fat tail" win-condition in PROBLEM.md §4).
- **T4 (reward-free consistency).** Under Bradley–Terry, relate preference-based move value to RM-based
  value. *Note:* the likelihood-smoke experiment already shows the DPO-margin verify is nearly
  uncorrelated with reward improvement (corr +0.10) — T4 should explain *why* (the margin is defined on
  two fixed completions, not the on-policy distribution), bounding when preference-value tracks reward.
