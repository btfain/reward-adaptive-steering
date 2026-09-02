"""P1 END-TO-END BANNER — the headline result. The FULLY-AUTOMATED pipeline (LLM-generated candidates ->
smoke-verify -> cluster -> cost-aware/greedy basis -> router trained on the candpool train swings -> top-k
at test) vs naive best-of-n on a HELD-OUT slice (disjoint from the router's training prompts), on the
reward-vs-#generations frontier. Ties (i)+(ii)+(iii) together into one figure.

  gen     generate n_base base + n_move per selected-move samples on the held-out prompts
  report  train the ridge router on candpool (features=distilroberta(train prompts), targets=swing[:,basis]),
          rank the K moves per held-out prompt, compute the frontier: router-move best-of-k vs naive BoN,
          rejection sampling, + oracle/random ranking bounds.

    python src/end_banner.py --phase gen  --base-config configs/base_7b.yaml --config configs/end_banner_7b.yaml
    python src/end_banner.py --phase report --config configs/end_banner_7b.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)
from bakeoff import _cfg, _emax_k
from bakeoff_rankers import embed, _ridge_rank

BASIS = REPO_ROOT / "basis"


def _out(c):
    d = REPO_ROOT / "results" / c["tag"]; d.mkdir(parents=True, exist_ok=True); return d


def _basis(c):
    s = json.load(open(REPO_ROOT / c["selection"]))
    K = c["K"]
    return [s["selected"][k]["move"] for k in range(K)], list(s["order"][:K])   # (move texts, candidate columns)


def _eval_prompts(c):
    allp = json.load(open(REPO_ROOT / "data" / "prompts.json"))[c["eval_split"]]
    return allp[c["eval_start"]:c["eval_start"] + c["eval_n"]]


def phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard):
    moves, _ = _basis(c); P = _eval_prompts(c)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": c["gen"]["max_new_tokens"], "do_sample": True,
        "temperature": c["gen"]["temperature"], "top_p": c["gen"]["top_p"]}}
    nb, nm = c["n_base"], c["n_move"]
    out = _out(c) / (f"gen_shard_{shard[0]}.jsonl" if shard else "gen.jsonl")
    with open(out, "w") as f:
        for pi, prompt in enumerate(P):
            if shard is not None and pi % shard[1] != shard[0]:
                continue
            for comp in generate_batch(model, tok, [prompt] * nb, gcfg):                 # base (no move)
                if comp.strip():
                    f.write(json.dumps({"pi": pi, "kind": "base", "move": -1,
                                        "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            for mj, mv in enumerate(moves):                                              # each selected move
                for comp in generate_batch(model, tok, [prompt] * nm, gcfg, system=mv):
                    if comp.strip():
                        f.write(json.dumps({"pi": pi, "kind": "move", "move": mj,
                                            "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            f.flush()
            if (pi + 1) % 10 == 0:
                print(f"  gen{'' if shard is None else ' shard '+str(shard[0])}: {pi+1}/{len(P)}", flush=True)
    print(f"gen -> {out}", flush=True)


def _boot_diff(d, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    b = np.sort([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(3000)])
    return float(np.mean(d)), float(b[75]), float(b[2925])


def phase_report(c):
    moves, cols = _basis(c); K = len(moves)
    O = _out(c); shards = sorted(O.glob("gen_shard_*.jsonl")) or [O / "gen.jsonl"]
    base = defaultdict(list); move = defaultdict(lambda: defaultdict(list))
    for sh in shards:
        for l in open(sh):
            r = json.loads(l)
            (base[r["pi"]].append(r["rm"]) if r["kind"] == "base" else move[r["pi"]][r["move"]].append(r["rm"]))
    P = _eval_prompts(c)
    pis = [pi for pi in range(len(P)) if len(base.get(pi, [])) >= 2 and len(move.get(pi, {})) == K]
    nb = min(len(base[pi]) for pi in pis)

    # ROUTER trained on candpool: distilroberta(train prompts) -> swing over the selected basis columns
    tr_prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[c["router_train_split"]][:c["router_train_n"]]
    Msel = np.load(REPO_ROOT / c["router_train_swing"], allow_pickle=True)["M"][:, cols]
    ok = ~np.isnan(Msel).all(1)
    Htr = embed(c["router_encoder"], tr_prompts)
    Xte = embed(c["router_encoder"], [P[pi] for pi in pis])
    rank = _ridge_rank(Htr[ok], np.nan_to_num(Msel[ok]), Xte)

    rng = np.random.default_rng(0); DRAWS = 400
    naive = np.zeros(K + 1); rmove = np.zeros(K + 1); omove = np.zeros(K + 1); rndm = np.zeros(K + 1)
    per_r2, per_n2 = [], []
    for ri, pi in enumerate(pis):
        b = np.array(base[pi]); bref = b.mean()
        sw = {mj: np.array(move[pi][mj]) - bref for mj in range(K)}
        orank = np.argsort(-np.array([sw[mj].mean() for mj in range(K)]))
        for k in range(1, K + 1):
            naive[k] += _emax_k(b, min(k, nb)) - bref
            for order, acc in ((rank[ri], rmove), (orank, omove)):
                acc[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in order[:k]) for _ in range(DRAWS)]))
            rndm[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in rng.permutation(K)[:k]) for _ in range(DRAWS)]))
        per_r2.append(np.mean([max(rng.choice(sw[mj]) for mj in rank[ri][:2]) for _ in range(DRAWS)]))
        per_n2.append(_emax_k(b, min(2, nb)) - bref)
    n = len(pis)
    for a in (naive, rmove, omove, rndm):
        a /= n
    rej = []
    for beta in (2, 1, 0.5, 0.25):
        g, r = [], []
        for pi in pis:
            b = np.array(base[pi]); w = np.exp((b - b.max()) / beta)
            g.append(1.0 / max(w.mean(), 1e-9)); r.append((w * b).sum() / w.sum() - b.mean())
        rej.append((beta, float(np.mean(g)), float(np.mean(r))))
    d2, lo2, hi2 = _boot_diff(np.array(per_r2) - np.array(per_n2))
    cross = next((k for k in range(1, K + 1) if naive[k] >= rmove[2]), None)

    rows = [f"# P1 END-TO-END BANNER — automated pipeline vs best-of-n (held-out) — {c['tag']}\n",
            f"{n} HELD-OUT prompts (disjoint from the {c['router_train_n']} router-training prompts). "
            f"Basis = candpool greedy top-{K}; router = ridge on candpool swings. Real best-of-k, ΔRM vs mean base.\n",
            "| k (gens) | naive BoN | **router-move (pipeline)** | move-oracle | random-move |", "|---|---|---|---|---|"]
    for k in range(1, K + 1):
        rows.append(f"| {k} | {naive[k]:+.3f} | {rmove[k]:+.3f} | {omove[k]:+.3f} | {rndm[k]:+.3f} |")
    rows += ["", "## Rejection sampling (β-swept)", "| β | exp #gens | reward |", "|---|---|---|"]
    for beta, g, r in rej:
        rows.append(f"| {beta:g} | {g:.2f} | {r:+.3f} |")
    rows += ["", "## Headline",
             f"- **pipeline router-move best-of-2 {rmove[2]:+.3f} vs naive best-of-2 {naive[2]:+.3f}: "
             f"Δ{d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]** ⇒ "
             + ("the FULLY-AUTOMATED pipeline BEATS best-of-n at equal compute on held-out prompts."
                if lo2 > 0 else "pipeline ≈/below naive — the end-to-end claim is not supported here."),
             f"- naive matches pipeline-2 at k={cross if cross else '>K'} (2 router-moves ≈ {cross} base samples).",
             f"- ranking headroom: random-2 {rndm[2]:+.3f} → router-2 {rmove[2]:+.3f} → oracle-2 {omove[2]:+.3f}."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / f"s1_{c['tag']}_report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print(f"report -> basis/s1_end_banner_{c['tag']}_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "report"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    args = ap.parse_args()
    c = _cfg(args.config)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    if args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg); t0 = time.time()
        model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
        phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard)
        print(log_cost("S1", "end_banner_gen", time.time() - t0, device, notes="end-to-end pipeline banner"))
    else:
        phase_report(c)


if __name__ == "__main__":
    main()
