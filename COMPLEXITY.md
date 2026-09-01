# Stage 1 — compute complexity profile (generations = the expensive unit)

A "generation" = one autoregressive rollout (~768 tokens). Backprop through the base LM is NEVER done
(base frozen); the router is a ~100M encoder (negligible vs the 7B). Parameters at the current empirical
scale: `C_raw≈2000` (raw candidates), `C≈200` (selected/medoid pool), `n_src=3`, `m_chk=m_sw=2`, `s≈48`
(subsampled prompts, T1), `N_tr≈1500`, `E≈9` (bandit epochs), `k≈2–3` (moves/query at test), `Q`=queries.

## Per-stage (generations)
| stage | generations | ≈ now | scales with | parallelism |
|---|---|---|---|---|
| (i) gen + smoke-verify | `C_raw·n_src·m_chk` (+ ~C_raw/ask_per gen calls) | ~12k | `C_raw` | embarrassingly parallel |
| (ii) swing matrix + greedy select | `C·s·m_sw`; select = free arithmetic | ~19k | `C·s` | embarrassingly parallel (cells) |
| (iii) bandit router training | `N_tr·E` | ~14k | `N_tr` | SEQUENTIAL in `E`; parallel within epoch |
| test-time (best-of-k) | `k` per query | `k·Q` | `Q` | parallel |

**Total one-time training ≈ 45k generations, no LM backprop.**

## Structure / bottleneck
- (i),(ii) are FIXED one-time costs (basis discovery; independent of deployment size) and embarrassingly
  parallel (wall-clock `/P`). (iii) is the only sequential stage (on-policy) and the only one scaling
  with `N_tr` ⇒ the wall-clock bottleneck at scale, but it is ~`E·N_tr/batch ≈ 420` cheap update-rounds
  (generate + tiny router step), NOT LM backprop.
- Optimizing (ii): cost is `C·s`. T1 handles `s` (prompt subsample, `s≪N`). The remaining lever is `C`
  (eliminate candidates before ever generating them = open best-arm problem, T1′). But the smoke test
  already prunes `C_raw 2000→200` upstream, so `C` is small by selection time ⇒ (ii) is near its floor.
- Greedy submodular itself contributes ~0 generations (arithmetic on the cached swing matrix); the
  "cost-aware" value is only in shrinking `s` (and, if solved, `C`).

## vs RLHF-PPO (against a frozen RM)
| | this pipeline | PPO |
|---|---|---|
| training samples | ~45k generations | ~1e5–1e6 rollout episodes |
| per-sample work | 1 frozen-LM forward | rollout + policy fwd+bwd + ref fwd + RM fwd + value fwd+bwd |
| LM backprop | none (frozen base) | YES through 7B+ (dominant cost) |
| memory | base LM + small RM + tiny router | 4 model copies + 7B optimizer states |
| parallelism | (i)/(ii) parallel; (iii) mildly sequential | fully sequential |
| test-time | `k×` (best-of-k) | `1×` (aligned model) |
| new RM/objective | re-run cheap basis discovery (or reuse basis) | full retrain |

Per episode ≈ 6–8 generation-equivalents (policy fwd+bwd ≈3×, + ref/RM/value). ⇒ 100k-episode PPO ≈
~700k gen-equiv (~15× ours); 1M ≈ ~7M (~150×) — plus 7B backprop, 4× memory, fully sequential.

**Placement:** PPO front-loads a huge sequential LM-training cost, then inference is `1×`. We keep a small
parallel frozen-base training cost and pay `k×` at test. Rough total-compute crossover at
`Q ≈ (C_ppo−C_ours)/(k−1) ≈ few-million queries`. Beyond that PPO amortizes better on pure compute; below
it we are cheaper — and we always keep the frozen/interpretable base, no training infra, and instant reuse
across any RM/verifier.


## Memory & communication (single-GPU vs multi-GPU) — the democratization point
PPO on a 7B policy: weights 14GB + grads 14GB + Adam(fp32 m,v) ~56GB + activations ≈ **~85GB for the
trainable policy alone**, plus reference (~14GB) + reward (7B RM ~14GB) + value/critic (~85GB if trained)
≈ **~150–200GB** ⇒ *full-parameter* PPO REQUIRES a multi-GPU A100/H100 cluster (ZeRO / model-parallel,
communication-bound). **LoRA/QLoRA-PPO DOES fit on one 24GB GPU** (frozen 4-bit base + LoRA + value head +
0.6B RM ≈ 6–8GB w/ grad-checkpointing) — but it still **backprops through the full 7B forward graph**
(stores 7B activations, 7B-wide backward), which we avoid ENTIRELY (forward-only generation). So the honest
claim is 'no LM backprop + ~1000× fewer samples', not 'PPO cannot run here'.
Ours: base LM **frozen, inference-only** (14GB bf16, ~4GB 4-bit; NO grads/optimizer for the 7B) + RM 0.6B
(~1.2GB) + router (~100M, <1GB) ⇒ **fits on ONE 24GB A5000 — the hardware this whole project ran on.**
Parallel path is communication-FREE: (i)/(ii) are independent generations; (iii)'s only trained object is
a tiny UNSHARDED router ⇒ no 7B gradient-sync. So vs PPO it is not "less communication" but ~none on the
expensive path, on commodity single-GPU hardware PPO-on-7B fundamentally cannot use.

## Router sample complexity (measured) — revises (iii) down
Held-out top-k value vs training-set size on the cached large_7b matrix (8 seeds, K=8):
`N_tr = 30/60/120/200/300 → top-2 = +0.66/+0.67/+0.62/+0.57/+0.59` — **plateaus by ~60 prompts; more data
does not help** (top-1 even drifts down: info-limited ceiling, extra data fits noise). So the router needs
`N_tr ≈ 60–100` (vs PPO ~1e5–1e6, 3–4 orders fewer) ⇒ **(iii) ≈ N_tr·E ≈ 100·9 ≈ 1k generations, the
CHEAPEST stage**, not the bottleneck. Whole-pipeline total tightens to ~30k gens, dominated by (i)+(ii).
Reason (hypothesis, OOD half untested → P3): we fit a K-way ranking via a small encoder, not a 7B policy —
tiny hypothesis class, semantically coarse map ("which kind of prompt wants which move") ⇒ generalizes
held-out (bake-off: trained on large_7b, beat naive on held-out b1). T2 = the formal generalization bound.