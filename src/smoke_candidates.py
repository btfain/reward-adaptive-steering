"""P1(i) verify step (sharded) — was the model's zero-shot GUESS correct? Each raw candidate is the
generator's hypothesis, from a specific prompt's high-vs-low contrast, that this move helps. Test it ON
ITS SOURCE prompts (via provenance) vs the CACHED base reward; cull wrong guesses (in-context swing ≤ 0).
FLOOR filter, NOT top-K (complementary moves that help their niche pass; average-ranking is valid only
for move #1). Sharded over the large raw pool for parallel generation.

  test  --shard i/N   score candidate slice on source prompts -> shard_i.jsonl {candidate, swing}
  assemble            merge shards, cull swing<=threshold -> verified candidates + provenance subset

    python src/smoke_candidates.py --phase test --shard 0/4 --base-config configs/base_7b.yaml
    python src/smoke_candidates.py --phase assemble
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)

BASIS = REPO_ROOT / "basis"


def _read(path):
    return [l.strip() for l in open(REPO_ROOT / path) if l.strip() and not l.startswith("#")]


def _out(tag):
    d = REPO_ROOT / "results" / f"smoke_candidates_{tag}"; d.mkdir(parents=True, exist_ok=True); return d


def phase_test(args, base_cfg, device):
    cands = _read(args.candidates)
    prov = json.load(open(REPO_ROOT / args.provenance))
    bagg = defaultdict(list)
    for l in open(REPO_ROOT / args.pool):
        r = json.loads(l)
        if r["completion"].strip():
            bagg[r["prompt"]].append(r["rm"])
    base_ref = {p: float(np.mean(v)) for p, v in bagg.items()}
    i, N = (int(x) for x in args.shard.split("/"))
    mine = [(gi, c) for gi, c in enumerate(cands) if gi % N == i]

    model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 768, "do_sample": True, "temperature": 0.9, "top_p": 0.95}}
    out = _out(args.tag) / f"shard_{i}.jsonl"
    with open(out, "w") as f:
        for n, (gi, cand) in enumerate(mine):
            srcs = [p for p in prov.get(cand, []) if p in base_ref]
            if not srcs:
                f.write(json.dumps({"candidate": cand, "swing": None}) + "\n"); continue
            flat = [p for p in srcs for _ in range(args.m_check)]
            comps = generate_batch(model, tok, flat, gcfg, system=cand)
            by = defaultdict(list)
            for p, c in zip(flat, comps):
                if c.strip():
                    by[p].append(rm_score(rm, rm_tok, p, c))
            sw = float(np.mean([np.mean(by[p]) - base_ref[p] for p in by])) if by else None
            f.write(json.dumps({"candidate": cand, "swing": sw}) + "\n"); f.flush()
            if (n + 1) % 25 == 0:
                print(f"  shard {i}: {n+1}/{len(mine)}", flush=True)
    print(f"smoke shard {i} -> {out}", flush=True)


def phase_assemble(args):
    prov = json.load(open(REPO_ROOT / args.provenance))
    rows = [json.loads(l) for f in sorted(_out(args.tag).glob("shard_*.jsonl")) for l in open(f)]
    kept, culled, untested, swings = [], [], [], {}
    for r in rows:
        c, sw = r["candidate"], r["swing"]
        swings[c] = sw
        if sw is None:
            kept.append(c); untested.append(c)                      # can't test -> keep (lenient)
        elif sw > args.threshold:
            kept.append(c)
        else:
            culled.append(c)
    (REPO_ROOT / args.out).write_text(
        "# SMOKE-VERIFIED raw candidates (zero-shot guess helped in-context on source prompts)\n"
        + "\n".join(kept) + "\n")
    json.dump({c: prov[c] for c in kept if c in prov}, open(REPO_ROOT / args.provenance_out, "w"))
    order = sorted(((c, s) for c, s in swings.items() if s is not None), key=lambda x: x[1])
    rep = [f"# P1(i) smoke test (in-context) — kept {len(kept)} ({len(untested)} untested-kept), "
           f"culled {len(culled)} of {len(rows)} raw candidates.\n",
           "## Culled (wrong guesses — should be junk/leaked/degenerate):"]
    rep += [f"- {s:+.3f}  {c}" for c, s in order if c in culled][:25]
    rep += ["", "## Top verified (sanity):"]
    rep += [f"- {s:+.3f}  {c}" for c, s in reversed(order) if c in kept][:12]
    BASIS.mkdir(exist_ok=True)
    (BASIS / f"s1_smoke_candidates_{args.tag}_report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\nverified {len(kept)}/{len(rows)} -> {args.out}  provenance -> {args.provenance_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["test", "assemble"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--candidates", default="configs/candidates_raw_7b.txt")
    ap.add_argument("--provenance", default="configs/candidates_raw_7b.provenance.json")
    ap.add_argument("--pool", default="results/prompt_basis_large_7b/pool.jsonl")
    ap.add_argument("--m_check", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out", default="configs/candidates_verified_7b.txt")
    ap.add_argument("--provenance_out", default="configs/candidates_verified_7b.provenance.json")
    args = ap.parse_args()
    if args.phase == "test":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
        t0 = time.time()
        phase_test(args, base_cfg, device)
        print(log_cost("S1", "smoke_candidates", time.time() - t0, device, notes="in-context zero-shot-guess verify"))
    else:
        phase_assemble(args)


if __name__ == "__main__":
    main()
