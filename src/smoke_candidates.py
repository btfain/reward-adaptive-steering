"""P1(i) verify step — smoke-test each auto candidate: was the model's zero-shot GUESS correct? Each
candidate is the generator's hypothesis, from a specific prompt's high-vs-low contrast, that this move
would help. We test it ON THOSE SOURCE PROMPTS (where it was believed helpful — NOT random prompts,
which would be an average-value screen that wrongly culls complementary moves). Generate a few
completions under the candidate on its source prompts, RM-score, compare to the CACHED base reward from
the pool (no base regeneration), and cull candidates whose in-context swing is clearly ≤ 0 (the wrong
guesses: junk / leaked / degenerate).

FLOOR filter (cull confidently-non-helpful), NOT top-K — complementary moves that help their niche pass
and go to (ii) for marginal-gain judgement. Cheap: Σ_candidate |source| × m_check gens.

    python src/smoke_candidates.py --base-config configs/base_7b.yaml --candidates configs/candidates_auto_7b.txt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--candidates", default="configs/candidates_auto_7b.txt")
    ap.add_argument("--provenance", default="configs/candidates_auto_7b.provenance.json")
    ap.add_argument("--pool", default="results/prompt_basis_large_7b/pool.jsonl", help="cached base rewards")
    ap.add_argument("--curated", default="configs/candidates_seed_v2.txt")
    ap.add_argument("--m_check", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0, help="cull if in-context swing <= this (floor)")
    ap.add_argument("--out", default="configs/candidates_auto_smoke_7b.txt")
    ap.add_argument("--combined_out", default="configs/candidates_combined_7b.txt")
    args = ap.parse_args()
    base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
    t0 = time.time()

    cands = _read(args.candidates); curated = _read(args.curated)
    prov = json.load(open(REPO_ROOT / args.provenance))                # candidate -> [source prompt texts]
    # cached base reward per source prompt (mean over base completions in the pool)
    bagg = defaultdict(list)
    for l in open(REPO_ROOT / args.pool):
        r = json.loads(l)
        if r["completion"].strip():
            bagg[r["prompt"]].append(r["rm"])
    base_ref = {p: float(np.mean(v)) for p, v in bagg.items()}

    model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 768, "do_sample": True, "temperature": 0.9, "top_p": 0.95}}

    swings, kept, culled, untested = {}, [], [], []
    for i, cand in enumerate(cands):
        srcs = [p for p in prov.get(cand, []) if p in base_ref]        # source prompts w/ cached base reward
        if not srcs:
            kept.append(cand); untested.append(cand); swings[cand] = float("nan")   # can't test -> keep (lenient)
            print(f"  [{i+1}/{len(cands)}] no source/base -> KEEP (untested): {cand[:60]}", flush=True); continue
        flat = [p for p in srcs for _ in range(args.m_check)]
        comps = generate_batch(model, tok, flat, gcfg, system=cand)
        by = defaultdict(list)
        for p, c in zip(flat, comps):
            if c.strip():
                by[p].append(rm_score(rm, rm_tok, p, c))
        sw = float(np.mean([np.mean(by[p]) - base_ref[p] for p in by]))   # in-context swing vs cached base
        swings[cand] = sw
        (kept if sw > args.threshold else culled).append(cand)
        print(f"  [{i+1}/{len(cands)}] in-context swing {sw:+.3f}  "
              f"{'KEEP' if sw > args.threshold else 'cull'}: {cand[:55]}", flush=True)

    (REPO_ROOT / args.out).write_text(
        "# auto candidates whose zero-shot guess VERIFIED (in-context swing>0 on source prompts)\n"
        + "\n".join(kept) + "\n")
    seen = {c.lower().rstrip(".") for c in kept}
    cur_new = [c for c in curated if c.lower().rstrip(".") not in seen]
    combo = kept + cur_new
    (REPO_ROOT / args.combined_out).write_text(
        f"# combined pool (smoke-verified auto + curated). n_auto={len(kept)}\n" + "\n".join(combo) + "\n")

    order = sorted(((c, s) for c, s in swings.items() if not np.isnan(s)), key=lambda x: x[1])
    rows = [f"# P1(i) smoke test — was the zero-shot guess correct? (in-context floor filter)\n",
            f"{len(cands)} auto candidates tested on THEIR SOURCE prompts (× {args.m_check}) vs cached base "
            f"reward. FLOOR filter (cull in-context swing ≤ {args.threshold}), NOT top-K. "
            f"Kept {len(kept)} ({len(untested)} untested-kept), culled {len(culled)}.\n",
            f"Combined pool: {len(kept)} verified auto + {len(cur_new)} curated = {len(combo)}.\n",
            "## Culled (wrong guesses — should be junk/leaked/degenerate):"]
    rows += [f"- {s:+.3f}  {c}" for c, s in order if c in culled][:20]
    rows += ["", "## Top verified (sanity — sensible general moves that helped in-context):"]
    rows += [f"- {s:+.3f}  {c}" for c, s in reversed(order) if c in kept][:10]
    BASIS.mkdir(exist_ok=True)
    (BASIS / "s1_smoke_candidates_report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nkept {len(kept)}/{len(cands)} -> {args.out}  (combined -> {args.combined_out})")
    print(log_cost("S1", "smoke_candidates", time.time() - t0, device, notes="in-context zero-shot-guess verify"))


if __name__ == "__main__":
    main()
