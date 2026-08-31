"""P1(a) — the standardized BANNER harness: the reward-vs-#generations frontier that every Stage-1
result reports into. On a generation dataset (per-prompt base samples + per-move samples, all RM-scored),
computes REAL best-of-k (max reward over k actual generations, symmetric) for:
  * naive best-of-k        — k base regenerations (the strong baseline)
  * rejection sampling      — accept base draws ∝ exp((r−r_max)/β); β-swept reward-vs-expected-#gens curve
  * router-move best-of-k   — 1 sample under each of the ranker's top-k moves (the METHOD)
  (+ move-oracle / random-move as upper/lower ranking bounds)
All ΔRM vs mean base. Headline: router-move-2 vs naive-2 (paired CI) + crossover k. Soft-BoN is dominated
on reward-vs-#gens (BoN=argmax is the reward-max selection); it belongs on the reward-vs-KL axis (future).

    python src/banner.py --config configs/bakeoff_7b.yaml
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from bakeoff import _cfg, _out, _prompts, _moves, _emax_k
from bakeoff_rankers import embed, _l7, _ridge_rank, REPO_ROOT


def load_gens(c):
    O = _out(c)
    shards = sorted(O.glob("gen_shard_*.jsonl")) or [O / "gen.jsonl"]
    base = defaultdict(list); move = defaultdict(lambda: defaultdict(list))
    for sh in shards:
        for l in open(sh):
            r = json.loads(l)
            (base[r["pi"]].append(r["rm"]) if r["kind"] == "base" else move[r["pi"]][r["move"]].append(r["rm"]))
    return base, move


def _boot_diff(d, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    b = np.sort([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(3000)])
    return float(np.mean(d)), float(b[75]), float(b[2925])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bakeoff_7b.yaml")
    ap.add_argument("--betas", default="2,1,0.5,0.25,0.1")
    args = ap.parse_args()
    c = _cfg(args.config); base, move = load_gens(c)
    P = _prompts(c); K = len(_moves(c))
    pis = [pi for pi in range(len(P)) if len(base.get(pi, [])) >= 2 and len(move.get(pi, {})) == K]
    nb = min(len(base[pi]) for pi in pis)

    # ranker: ridge on large_7b (validated ≈ bandit-as-ranker), applied to these held-out prompts
    Ptr, Ytr = _l7(c)
    Xte = embed(c["router_encoder"], [P[pi] for pi in pis])
    Htr = np.load(REPO_ROOT / c["router_enc_embed"], allow_pickle=True)["Htr"]
    ok = ~np.isnan(np.load(REPO_ROOT / c["router_swing"], allow_pickle=True)["M"]).all(1)
    rank = _ridge_rank(Htr[ok], Ytr, Xte)

    rng = np.random.default_rng(0); DRAWS = 400
    naive = np.zeros(K + 1); rmove = np.zeros(K + 1); omove = np.zeros(K + 1); rndm = np.zeros(K + 1)
    per_router2, per_naive2 = [], []
    for r_i, pi in enumerate(pis):
        b = np.array(base[pi]); bref = b.mean()
        sw = {mj: np.array(move[pi][mj]) - bref for mj in range(K)}
        orank = np.argsort(-np.array([sw[mj].mean() for mj in range(K)]))
        for k in range(1, K + 1):
            naive[k] += _emax_k(b, min(k, nb)) - bref
            for order, acc in ((rank[r_i], rmove), (orank, omove)):
                acc[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in order[:k]) for _ in range(DRAWS)]))
            rndm[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in rng.permutation(K)[:k]) for _ in range(DRAWS)]))
        per_router2.append(np.mean([max(rng.choice(sw[mj]) for mj in rank[r_i][:2]) for _ in range(DRAWS)]))
        per_naive2.append(_emax_k(b, min(2, nb)) - bref)
    n = len(pis)
    for a in (naive, rmove, omove, rndm):
        a /= n

    # rejection sampling frontier (β-swept): accept base draw ∝ exp((r−rmax)/β)
    rej = []
    for beta in [float(x) for x in args.betas.split(",")]:
        gens, rew = [], []
        for pi in pis:
            b = np.array(base[pi]); w = np.exp((b - b.max()) / beta)
            gens.append(1.0 / max(w.mean(), 1e-9)); rew.append((w * b).sum() / w.sum() - b.mean())
        rej.append((beta, float(np.mean(gens)), float(np.mean(rew))))

    d2, lo2, hi2 = _boot_diff(np.array(per_router2) - np.array(per_naive2))
    cross = next((k for k in range(1, K + 1) if naive[k] >= rmove[2]), None)
    rows = [f"# BANNER — reward vs #generations frontier — {c['tag']}\n",
            f"{n} held-out prompts, real best-of-k (max reward over k actual gens). ΔRM vs mean base. "
            f"Ranker = ridge (≈ validated bandit-as-ranker).\n",
            "| k (gens) | naive BoN | **router-move** | move-oracle | random-move |", "|---|---|---|---|---|"]
    for k in range(1, K + 1):
        rows.append(f"| {k} | {naive[k]:+.3f} | {rmove[k]:+.3f} | {omove[k]:+.3f} | {rndm[k]:+.3f} |")
    rows += ["", "## Rejection sampling (β-swept: expected #gens → reward)",
             "| β | exp. #gens | reward ΔRM |", "|---|---|---|"]
    for beta, g, r in rej:
        rows.append(f"| {beta:g} | {g:.2f} | {r:+.3f} |")
    rows += ["", "## Headline",
             f"- **router-move best-of-2 {rmove[2]:+.3f} vs naive best-of-2 {naive[2]:+.3f}: "
             f"Δ{d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]**  ⇒ "
             + ("router-narrowed selection BEATS naive resampling at equal compute." if lo2 > 0 else
                "router-move ≈/below naive — cost story not supported here."),
             f"- naive matches router-move-2 at k={cross if cross else '>K'} (2 moves ≈ {cross} base samples).",
             f"- router captures ranking headroom random-2 {rndm[2]:+.3f} → router-2 {rmove[2]:+.3f} → oracle-2 {omove[2]:+.3f}.",
             "- soft-BoN omitted (dominated on reward-vs-#gens; BoN=argmax is the reward-max selection) — it "
             "belongs on the reward-vs-KL axis (future extension)."]
    rpt = REPO_ROOT / "basis" / f"s1_banner_{c['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print(f"report -> {rpt}")


if __name__ == "__main__":
    main()
