"""P1(i) verify step — smoke-test each auto-generated candidate: does it actually push the base model
toward higher reward? Generate a few completions under the candidate on a small DISJOINT check set,
RM-score, and cull candidates whose mean swing vs base is clearly ≤ 0 (junk / leaked / degenerate — e.g.
the 'from Vitable' leak, task-specific moves that hurt off-domain).

IMPORTANT: this is a FLOOR filter (cull clearly-non-helpful), NOT a top-K screen. A candidate with modest
AVERAGE swing can still be highly COMPLEMENTARY (helps a subset nothing else covers); average-ranking is
valid only for move #1 in submodular selection. So we keep everything non-negative for (ii) to judge on
marginal gain, and only drop the confidently-negative. Check prompts are disjoint from the eval/reference
set (no selection bias). Cheap: n_check × (1 + C) × m_check gens.

    python src/smoke_candidates.py --base-config configs/base_7b.yaml --candidates configs/candidates_auto_7b.txt
"""

import argparse
import json
import time
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
    ap.add_argument("--curated", default="configs/candidates_seed_v2.txt")
    ap.add_argument("--split", default="train_large")
    ap.add_argument("--prompt_start", type=int, default=96, help="DISJOINT from the reference set [0:96]")
    ap.add_argument("--n_check", type=int, default=8)
    ap.add_argument("--m_check", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.0, help="cull if mean swing <= this (floor, not top-K)")
    ap.add_argument("--out", default="configs/candidates_auto_smoke_7b.txt")
    ap.add_argument("--combined_out", default="configs/candidates_combined_7b.txt")
    args = ap.parse_args()
    base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
    t0 = time.time()

    cands = _read(args.candidates); curated = _read(args.curated)
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[args.split][
        args.prompt_start:args.prompt_start + args.n_check]
    model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 768, "do_sample": True, "temperature": 0.9, "top_p": 0.95}}

    def score(system):                                              # mean RM over n_check×m_check gens
        flat = [p for p in prompts for _ in range(args.m_check)]
        comps = generate_batch(model, tok, flat, gcfg, system=system)
        r = {}
        for pi, (p, c) in enumerate(zip(flat, comps)):
            if c.strip():
                r.setdefault(p, []).append(rm_score(rm, rm_tok, p, c))
        return {p: float(np.mean(v)) for p, v in r.items()}

    base_r = score(None)
    swings, kept, culled = {}, [], []
    for i, cand in enumerate(cands):
        sr = score(cand)
        sw = float(np.mean([sr[p] - base_r[p] for p in sr if p in base_r]))
        swings[cand] = sw
        (kept if sw > args.threshold else culled).append(cand)
        print(f"  [{i+1}/{len(cands)}] swing {sw:+.3f}  {'KEEP' if sw > args.threshold else 'cull'}: {cand[:60]}", flush=True)

    outp = REPO_ROOT / args.out
    outp.write_text("# auto candidates that PASSED the smoke test (swing>0 on disjoint check set)\n" + "\n".join(kept) + "\n")
    seen = {c.lower().rstrip(".") for c in kept}
    cur_new = [c for c in curated if c.lower().rstrip(".") not in seen]
    combo = kept + cur_new
    (REPO_ROOT / args.combined_out).write_text(
        f"# combined pool (smoke-verified auto + curated). n_auto={len(kept)}\n" + "\n".join(combo) + "\n")

    order = sorted(swings.items(), key=lambda x: x[1])
    rows = [f"# P1(i) smoke test — candidate reward-improvement floor filter\n",
            f"{len(cands)} auto candidates, {args.n_check} disjoint check prompts × {args.m_check} samples. "
            f"FLOOR filter (cull swing ≤ {args.threshold}), NOT top-K. Kept {len(kept)}, culled {len(culled)}.\n",
            f"Combined pool: {len(kept)} verified auto + {len(cur_new)} curated = {len(combo)}.\n",
            "## Culled (lowest swing — should be junk/leaked/degenerate):"]
    rows += [f"- {sw:+.3f}  {c}" for c, sw in order if c in culled][:20]
    rows += ["", "## Top-swing kept (sanity — should be sensible general moves):"]
    rows += [f"- {sw:+.3f}  {c}" for c, sw in reversed(order) if c in kept][:10]
    (BASIS).mkdir(exist_ok=True)
    (BASIS / "s1_smoke_candidates_report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nkept {len(kept)}/{len(cands)} -> {outp}  (combined -> {args.combined_out})")
    print(log_cost("S1", "smoke_candidates", time.time() - t0, device, notes="candidate reward-improvement floor filter"))


if __name__ == "__main__":
    main()
