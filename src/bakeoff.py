"""Selection bake-off — router-narrowed move selection vs naive best-of-n, REAL best-of-k at equal
compute. The decisive cost experiment (offline data couldn't settle it: move numbers were mean-based
and cross-set). Here we generate, on ONE held-out set, per-sample BASE gens and per-sample MOVE gens
(under each discovered move), record every RM, then compute the DEPLOYED quantity — max over k actual
generations — for both, symmetrically (no winner's-curse asymmetry):

  naive best-of-k        = max RM over k base regenerations           (no basis/router; the real bar)
  router-move best-of-k  = max RM over 1 sample under each router top-k move
  (+ move-oracle best-of-k = rank by TRUE mean swing = router upper bound; random-k = router lower bound)

Headline: does router-move-best-of-2 beat naive-best-of-2? at what k does naive catch up? Answers whether
a discovered reward-driven basis + a cheap router earns its keep against dead-simple resampling.

Phases: gen (--shard i/N, GPU, sharded) -> report (CPU: assemble + ridge router + best-of-k curves).

    python src/bakeoff.py --phase gen --shard 0/3 --base-config configs/base_7b.yaml --config configs/bakeoff_7b.yaml
    python src/bakeoff.py --phase report --config configs/bakeoff_7b.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import yaml

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)

BASIS = REPO_ROOT / "basis"


def _cfg(path):
    with open(REPO_ROOT / path) as f:
        return yaml.safe_load(f)


def _out(c):
    d = REPO_ROOT / "results" / f"bakeoff_{c['tag']}" if not c["tag"].startswith("bakeoff") \
        else REPO_ROOT / "results" / c["tag"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prompts(c):
    allp = json.load(open(REPO_ROOT / "data" / "prompts.json"))[c["prompts_split"]]
    s = c["prompt_start"]
    return allp[s:s + c["n_prompts"]]


def _moves(c):
    return [x["move"] for x in json.load(open(REPO_ROOT / c["basis_selection"]))["selected"]]


def _gcfg(base_cfg, c):
    return {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": c["gen"]["max_new_tokens"], "do_sample": True,
        "temperature": c["gen"]["temperature"], "top_p": c["gen"]["top_p"]}}


# --------------------------------------------------------------------- gen ----
def phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard):
    P = _prompts(c); moves = _moves(c); gcfg = _gcfg(base_cfg, c)
    nb, nm = c["n_base"], c["n_move"]
    import torch
    torch.manual_seed(c["seed"] + (0 if shard is None else shard[0] + 1))
    out = _out(c) / (f"gen_shard_{shard[0]}.jsonl" if shard else "gen.jsonl")
    with open(out, "w") as f:
        for pi, prompt in enumerate(P):
            if shard is not None and pi % shard[1] != shard[0]:
                continue
            for comp in generate_batch(model, tok, [prompt] * nb, gcfg):        # base (no system)
                if comp.strip():
                    f.write(json.dumps({"pi": pi, "kind": "base", "move": -1,
                                        "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            for mj, mv in enumerate(moves):                                     # each move
                for comp in generate_batch(model, tok, [prompt] * nm, gcfg, system=mv):
                    if comp.strip():
                        f.write(json.dumps({"pi": pi, "kind": "move", "move": mj,
                                            "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            f.flush()
            if (pi + 1) % 10 == 0:
                print(f"  gen{'' if shard is None else ' shard '+str(shard[0])}: {pi+1}/{len(P)} prompts", flush=True)
    print(f"gen -> {out}", flush=True)


# ------------------------------------------------------------------ report ----
def _emax_k(rs, k):                       # E[max of random k-subset without replacement], order stats
    rs = np.sort(np.asarray(rs, float)); n = len(rs)
    if k > n:
        return float(rs.mean())
    den = comb(n, k)
    return float(sum(rs[i] * comb(i, k - 1) / den for i in range(k - 1, n)))


def _ridge_router(c):
    """Ridge encoder->swing on large_7b; returns a function prompt-embeddings -> predicted swing (rank)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    ec = np.load(REPO_ROOT / c["router_enc_embed"], allow_pickle=True); Htr = ec["Htr"]
    S = json.load(open(REPO_ROOT / c["basis_selection"].replace("selection.json", "selection.json")))["order"]
    M = np.load(REPO_ROOT / c["router_swing"], allow_pickle=True)["M"][:, S]
    ok = ~np.isnan(M).all(1)
    X, Y = Htr[ok], np.nan_to_num(M[ok])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd; dmu = Y.mean(0)
    W = np.linalg.solve(Xn.T @ Xn + 1.0 * np.eye(Xn.shape[1]), Xn.T @ (Y - dmu))
    tok = AutoTokenizer.from_pretrained(c["router_encoder"])
    mdl = AutoModel.from_pretrained(c["router_encoder"]).eval()

    def rank(prompts):
        embs = []
        with torch.no_grad():
            for s in range(0, len(prompts), 16):
                e = tok(prompts[s:s + 16], padding=True, truncation=True, max_length=160, return_tensors="pt")
                h = mdl(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).float()
                embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
        E = np.concatenate(embs)
        pred = (E - mu) / sd @ W + dmu
        return np.argsort(-pred, axis=1)                                        # best-predicted move first
    return rank


def phase_report(c):
    O = _out(c)
    shards = sorted(O.glob("gen_shard_*.jsonl")) or [O / "gen.jsonl"]
    base = defaultdict(list); move = defaultdict(lambda: defaultdict(list))
    for sh in shards:
        for l in open(sh):
            r = json.loads(l)
            if r["kind"] == "base":
                base[r["pi"]].append(r["rm"])
            else:
                move[r["pi"]][r["move"]].append(r["rm"])
    P = _prompts(c); moves = _moves(c); K = len(moves)
    pis = [pi for pi in range(len(P)) if len(base.get(pi, [])) >= 2 and len(move.get(pi, {})) == K]
    rank = _ridge_router(c)([P[pi] for pi in pis])                             # (n, K) router move ranking
    nb = min(len(base[pi]) for pi in pis)

    rng = np.random.default_rng(0); DRAWS = 400
    naive = np.zeros(K + 1); rmove = np.zeros(K + 1); omove = np.zeros(K + 1); rndm = np.zeros(K + 1)
    for r_i, pi in enumerate(pis):
        b = np.array(base[pi]); bref = b.mean()
        sw = {mj: np.array(move[pi][mj]) - bref for mj in range(K)}            # per-sample swing per move
        mean_sw = np.array([sw[mj].mean() for mj in range(K)])
        oracle_rank = np.argsort(-mean_sw)                                     # rank by TRUE mean swing
        for k in range(1, K + 1):
            naive[k] += _emax_k(b, min(k, nb)) - bref                          # naive best-of-k base
            for order, acc in ((rank[r_i], rmove), (oracle_rank, omove)):
                top = order[:k]
                draws = [max(rng.choice(sw[mj]) for mj in top) for _ in range(DRAWS)]  # 1 sample/move, max
                acc[k] += float(np.mean(draws))
            rk = rng.permutation(K)[:k]
            rndm[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in rk) for _ in range(DRAWS)]))
    n = len(pis)
    for a in (naive, rmove, omove, rndm):
        a /= n

    def boot_diff(kk):                                                          # per-prompt paired CI: router-move[k] - naive[k]
        d = []
        for pi in pis:
            b = np.array(base[pi]); bref = b.mean()
            sw = {mj: np.array(move[pi][mj]) - bref for mj in range(K)}
            top = rank[pis.index(pi)][:kk]
            rm_ = np.mean([max(rng.choice(sw[mj]) for mj in top) for _ in range(200)])
            d.append(rm_ - (_emax_k(b, min(kk, nb)) - bref))
        d = np.array(d); bs = np.sort([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)])
        return float(np.mean(d)), float(bs[50]), float(bs[1950])

    d2m, d2lo, d2hi = boot_diff(2)
    # crossover: smallest naive k reaching router-move best-of-2
    cross = next((k for k in range(1, K + 1) if naive[k] >= rmove[2]), None)
    rows = [f"# Selection bake-off — router-move vs naive best-of-k (REAL, equal compute) — {c['tag']}\n",
            f"{n} held-out prompts, {nb} base + {c['n_move']}/move gens, K={K} moves, 768 tok. ΔRM vs mean base. "
            f"Both = max RM over k real generations (symmetric). Router = ridge (large_7b distilroberta→swing).\n",
            "| k (gens) | naive best-of-k | router-move best-of-k | move-oracle | random-move |",
            "|---|---|---|---|---|"]
    for k in range(1, K + 1):
        rows.append(f"| {k} | {naive[k]:+.3f} | {rmove[k]:+.3f} | {omove[k]:+.3f} | {rndm[k]:+.3f} |")
    rows += ["", "## Reading",
             f"- **router-move best-of-2 {rmove[2]:+.3f} vs naive best-of-2 {naive[2]:+.3f}: "
             f"Δ{d2m:+.3f} [{d2lo:+.3f}, {d2hi:+.3f}]**"
             + ("  ⇒ **the basis+router BEATS naive resampling at equal compute** — the cost story holds; "
                "a few reward-driven moves > many base samples."
                if d2lo > 0 else
                "  ⇒ naive resampling ≈/> router-move at k=2 ⇒ the discovered basis does NOT beat dead-simple "
                "best-of-n per generation ⇒ honest negative on the cost story."),
             f"- naive catches router-move-best-of-2 at k={cross if cross else '>K'} "
             + (f"(so 2 moves ≈ {cross} base samples)." if cross else "(naive never reaches it within K)."),
             f"- router vs its bounds: random-move-2 {rndm[2]:+.3f} (lower) ≤ router {rmove[2]:+.3f} ≤ "
             f"oracle {omove[2]:+.3f} (upper) — gap to oracle = router ranking headroom."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_{c['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "report"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    args = ap.parse_args()
    c = _cfg(args.config)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    t0 = time.time()
    if args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
        model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
        phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard)
        print(log_cost("S1", f"bakeoff_gen", time.time() - t0, device, notes="selection bake-off generation"))
    else:
        phase_report(c)


if __name__ == "__main__":
    main()
