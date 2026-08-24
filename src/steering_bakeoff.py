"""Step 2 — steering-by-selection bake-off. The fair test of the steering pushback: even though the
per-prompt optimal steering delta is NOT predictable (encoder-controller probe was null), maybe a small
reward-driven basis of steering DIRECTIONS has enough COVERAGE that best-of-k over directions (try each,
keep reward-best) beats naive best-of-k — the selection escape, for steering. Parallel to bakeoff.py
(same held-out prompts, same real best-of-k), so moves vs steering directions compare directly.

Direction basis = k-means centroids of the steer_reach per-prompt optima (deltas.npz D), each scaled to
the reach cap magnitude, applied as activation steering at the base steer_layer (alpha=1.0). Prior is
pessimistic (8 dirs ~ 8% of reachable-delta variance = high-rank wall), but this is the clean test.

Phases: build (CPU, once -> directions.npz) -> gen (--shard i/N, GPU) -> report (CPU).

    python src/steering_bakeoff.py --phase build  --config configs/steering_bakeoff_7b.yaml
    python src/steering_bakeoff.py --phase gen --shard 0/3 --base-config configs/base_7b.yaml --config configs/steering_bakeoff_7b.yaml
    python src/steering_bakeoff.py --phase report --config configs/steering_bakeoff_7b.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)
from bakeoff import _cfg, _prompts, _emax_k

BASIS = REPO_ROOT / "basis"


def _out(c):                                          # clean results dir (bakeoff._out mangles the name)
    d = REPO_ROOT / "results" / c["tag"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gcfg(base_cfg, c):
    return {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": c["gen"]["max_new_tokens"], "do_sample": True,
        "temperature": c["gen"]["temperature"], "top_p": c["gen"]["top_p"]}}


# ------------------------------------------------------------------- build ----
def phase_build(c):
    """k-means centroids of the reward-driven per-prompt optima -> K direction basis at cap magnitude."""
    z = np.load(REPO_ROOT / c["deltas"], allow_pickle=True)
    D = z["D"]; K = c["K_dirs"]
    norms = np.linalg.norm(D, axis=1); mag = float(np.median(norms))
    U = D / (norms[:, None] + 1e-9)                                   # unit directions
    rng = np.random.default_rng(c["seed"])
    C = U[rng.choice(len(U), K, replace=False)].copy()               # k-means (cosine) init
    for _ in range(50):
        assign = np.argmax(U @ C.T, axis=1)
        newC = np.array([U[assign == j].mean(0) if (assign == j).any() else C[j] for j in range(K)])
        newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9)
        if np.allclose(newC, C, atol=1e-5):
            C = newC; break
        C = newC
    dirs = C * mag                                                    # scale each centroid to cap magnitude
    np.savez(_out(c) / "directions.npz", dirs=dirs.astype(np.float32), mag=mag)
    sizes = [int((np.argmax(U @ C.T, 1) == j).sum()) for j in range(K)]
    print(f"built {K} steering directions (||.||={mag:.1f}), cluster sizes {sizes} -> {_out(c)/'directions.npz'}")


# --------------------------------------------------------------------- gen ----
def phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard):
    P = _prompts(c); gcfg = _gcfg(base_cfg, c)
    dirs = np.load(_out(c) / "directions.npz")["dirs"]; K = len(dirs)
    nb, nm = c["n_base"], c["n_move"]; alpha = c["alpha"]
    torch.manual_seed(c["seed"] + (0 if shard is None else shard[0] + 1))
    dvecs = [torch.tensor(dirs[j], dtype=torch.float32) for j in range(K)]
    out = _out(c) / (f"gen_shard_{shard[0]}.jsonl" if shard else "gen.jsonl")
    with open(out, "w") as f:
        for pi, prompt in enumerate(P):
            if shard is not None and pi % shard[1] != shard[0]:
                continue
            for comp in generate_batch(model, tok, [prompt] * nb, gcfg):        # base (no steering)
                if comp.strip():
                    f.write(json.dumps({"pi": pi, "kind": "base", "move": -1,
                                        "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            for dj in range(K):                                                 # each steering direction
                for comp in generate_batch(model, tok, [prompt] * nm, gcfg, vector=dvecs[dj], alpha=alpha):
                    if comp.strip():
                        f.write(json.dumps({"pi": pi, "kind": "dir", "move": dj,
                                            "rm": rm_score(rm, rm_tok, prompt, comp)}) + "\n")
            f.flush()
            if (pi + 1) % 10 == 0:
                print(f"  gen{'' if shard is None else ' shard '+str(shard[0])}: {pi+1}/{len(P)}", flush=True)
    print(f"gen -> {out}", flush=True)


# ------------------------------------------------------------------ report ----
def phase_report(c):
    O = _out(c)
    shards = sorted(O.glob("gen_shard_*.jsonl")) or [O / "gen.jsonl"]
    base = defaultdict(list); dd = defaultdict(lambda: defaultdict(list))
    for sh in shards:
        for l in open(sh):
            r = json.loads(l)
            (base[r["pi"]].append(r["rm"]) if r["kind"] == "base" else dd[r["pi"]][r["move"]].append(r["rm"]))
    K = len(np.load(O / "directions.npz")["dirs"])
    P = _prompts(c)
    pis = [pi for pi in range(len(P)) if len(base.get(pi, [])) >= 2 and len(dd.get(pi, {})) == K]
    nb = min(len(base[pi]) for pi in pis) if pis else 0

    rng = np.random.default_rng(0); DRAWS = 400
    naive = np.zeros(K + 1); orac = np.zeros(K + 1); rndm = np.zeros(K + 1)
    for pi in pis:
        b = np.array(base[pi]); bref = b.mean()
        sw = {dj: np.array(dd[pi][dj]) - bref for dj in range(K)}             # per-sample steering swing
        oracle_rank = np.argsort(-np.array([sw[dj].mean() for dj in range(K)]))
        for k in range(1, K + 1):
            naive[k] += _emax_k(b, min(k, nb)) - bref
            orac[k] += float(np.mean([max(rng.choice(sw[dj]) for dj in oracle_rank[:k]) for _ in range(DRAWS)]))
            rndm[k] += float(np.mean([max(rng.choice(sw[dj]) for dj in rng.permutation(K)[:k]) for _ in range(DRAWS)]))
    n = len(pis)
    for a in (naive, orac, rndm):
        a /= n

    # per-prompt paired CI: steering-ORACLE best-of-2 - naive best-of-2 (the coverage upper bound vs naive)
    d = []
    for pi in pis:
        b = np.array(base[pi]); bref = b.mean(); sw = {dj: np.array(dd[pi][dj]) - bref for dj in range(K)}
        orank = np.argsort(-np.array([sw[dj].mean() for dj in range(K)]))
        o2 = np.mean([max(rng.choice(sw[dj]) for dj in orank[:2]) for _ in range(200)])
        d.append(o2 - (_emax_k(b, min(2, nb)) - bref))
    d = np.array(d); bs = np.sort([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)])
    dlo, dhi = float(bs[50]), float(bs[1950])

    rows = [f"# Steering-by-selection bake-off — {c['tag']} (REAL best-of-k, {n} held-out prompts)\n",
            f"K={K} reward-driven steering directions (k-means of steer_reach optima) vs naive best-of-k, same "
            f"prompts as the MOVE bake-off. ΔRM vs mean base. steering-ORACLE = coverage upper bound (rank dirs "
            f"by true swing); if oracle ≈ naive, no router can rescue it.\n",
            "| k (gens) | naive best-of-k | steering-oracle | steering-random |", "|---|---|---|---|"]
    for k in range(1, K + 1):
        rows.append(f"| {k} | {naive[k]:+.3f} | {orac[k]:+.3f} | {rndm[k]:+.3f} |")
    rows += ["", "## Reading",
             f"- **steering-oracle best-of-2 {orac[2]:+.3f} vs naive best-of-2 {naive[2]:+.3f}: "
             f"Δ{d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]** (this is the UPPER bound — perfect router).",
             ("- steering directions HAVE coverage (oracle ≫ naive) ⇒ selection escape works for steering too; "
              "worth building the direction router (parallel to moves)."
              if dlo > 0.05 else
              "- **steering-oracle ≈/below naive ⇒ the direction basis does NOT cover reward-optimal regions better "
              "than plain resampling ⇒ steering-by-selection is dead even with a perfect router** — the high-rank "
              "wall (8 dirs ≈ 8% of reachable-δ variance) kills coverage. Confirms steering is out on ALL fronts."),
             "- Compare to the MOVE bake-off (s1_bakeoff_7b): moves oracle-2 ≈ +1.53 ≫ naive; if steering-oracle-2 "
             "is far below that, procedural moves cover reward-optimal regions that steering directions cannot."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_{c['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["build", "gen", "report"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    args = ap.parse_args()
    c = _cfg(args.config)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    t0 = time.time()
    if args.phase == "build":
        phase_build(c)
    elif args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
        model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
        phase_gen(base_cfg, c, model, tok, rm, rm_tok, shard)
        print(log_cost("S1", "steering_bakeoff_gen", time.time() - t0, device, notes="steering-by-selection bake-off"))
    else:
        phase_report(c)


if __name__ == "__main__":
    main()
