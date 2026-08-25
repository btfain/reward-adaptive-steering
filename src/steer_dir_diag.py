"""Diagnostic for the anomalous steering bake-off (steering-oracle = -6, catastrophic). Distinguishes:
  (A) HARNESS BUG — does this generation path reproduce steer_reach's per-prompt δ result (~+0.49)?
      Uses the SAME reach prompts + their OWN fitted δ (deltas.npz D). If own-δ swing ≈ +0.5 → harness
      is correct → the centroid destruction is real. If own-δ is also catastrophic → application bug
      (layer/space/magnitude), and the whole steering_bakeoff is invalid.
  (B) MAGNITUDE — sweep α on a shared centroid direction: is it destructive at all α, or fluent-but-
      unhelpful at low α (the clean 'no coverage' regime)?
Saves COMPLETION TEXT so we can read whether steered outputs are garbage vs fluent.

    python src/steer_dir_diag.py --base-config configs/base_7b.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)
from steer_cond import _prompts

BASIS = REPO_ROOT / "basis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--deltas", default="results/steer_reach_detrunc_7b/deltas.npz")
    ap.add_argument("--directions", default="results/steering_bakeoff_7b/directions.npz")
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=50)
    args = ap.parse_args()
    base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
    t0 = time.time()

    z = np.load(REPO_ROOT / args.deltas, allow_pickle=True); D = z["D"]
    dirs = np.load(REPO_ROOT / args.directions)["dirs"]
    P = _prompts(args.n_train, args.n_test)
    prompts = list(P["train"]) + list(P["test"])                     # order matches D rows
    assert len(prompts) == len(D), f"{len(prompts)} vs {len(D)}"
    idx = np.linspace(0, len(prompts) - 1, args.n_prompts).astype(int)   # spread across the set

    model, tok = load_base(base_cfg, device); rm, rm_tok = load_rm(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 256, "do_sample": True, "temperature": 0.9, "top_p": 0.95}}

    def gen(prompt, vec, alpha, m):
        v = None if vec is None else torch.tensor(vec, dtype=torch.float32)
        comps = generate_batch(model, tok, [prompt] * m, gcfg, vector=v, alpha=alpha)
        return [(c, rm_score(rm, rm_tok, prompt, c)) for c in comps if c.strip()]

    recs = []
    def add(pi, kind, alpha, res):
        for text, r in res:
            recs.append({"pi": int(pi), "kind": kind, "alpha": alpha, "rm": r, "text": text[:400]})

    for pi in idx:
        prompt = prompts[pi]
        add(pi, "base", 0.0, gen(prompt, None, 0.0, args.m))
        add(pi, "own_delta", 1.0, gen(prompt, D[pi], 1.0, args.m))            # SANITY: expect ~+0.49 vs base
        cj = int(np.argmax(dirs @ (D[pi] / (np.linalg.norm(D[pi]) + 1e-9)))) # nearest centroid to own δ
        for alpha in (0.25, 0.5, 1.0):                                       # MAGNITUDE sweep on a shared centroid
            add(pi, f"centroid{cj}", alpha, gen(prompt, dirs[cj], alpha, args.m))
        print(f"  diag {list(idx).index(pi)+1}/{len(idx)}", flush=True)

    O = REPO_ROOT / "results" / "steer_dir_diag"; O.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(O / "diag.json", "w"), indent=1)

    def swing(kind, alpha=None):
        by = {}
        for r in recs:
            by.setdefault(r["pi"], {}).setdefault((r["kind"], r["alpha"]), []).append(r["rm"])
        ds = []
        for pi, d in by.items():
            base = np.mean(d.get(("base", 0.0), [np.nan]))
            key = [k for k in d if k[0] == kind and (alpha is None or k[1] == alpha)]
            if key and not np.isnan(base):
                ds.append(np.mean(d[key[0]]) - base)
        return float(np.mean(ds)) if ds else float("nan")

    own = swing("own_delta")
    rows = [f"# Steering direction diagnostic — harness sanity + magnitude sweep ({len(idx)} prompts)\n",
            f"steer_layer={base_cfg['steer_layer']}, ‖δ‖≈{np.median(np.linalg.norm(D,axis=1)):.1f}. Swing = ΔRM vs base.\n",
            "## (A) Harness sanity — own fitted δ (should reproduce steer_reach ≈ +0.49)",
            f"- **own_delta swing = {own:+.3f}**  "
            + ("⇒ harness CORRECT (reproduces steer_reach) ⇒ centroid destruction is real."
               if own > 0.2 else
               ("⇒ own δ also weak/negative but not catastrophic ⇒ per-prompt δ doesn't reproduce; investigate."
                if own > -1.0 else
                "⇒ **own δ ALSO CATASTROPHIC ⇒ APPLICATION BUG (layer/space/magnitude) — steering_bakeoff INVALID**.")),
            "", "## (B) Magnitude sweep — shared centroid direction"]
    for alpha in (0.25, 0.5, 1.0):
        sw = swing(None, alpha) if False else np.mean([swing(k, alpha) for k in set(r["kind"] for r in recs if r["kind"].startswith("centroid"))])
        rows.append(f"- centroid @ α={alpha}: swing {sw:+.3f}")
    rows += ["", "## Reading",
             "- own δ ≈ +0.5 AND centroids negative at all α ⇒ no shared direction works (high-rank wall, "
             "clean) — steering-by-selection dead, rigorously.",
             "- own δ catastrophic ⇒ harness bug ⇒ FIX before concluding anything about steering selection.",
             "- centroids fluent (near 0) at low α but negative ⇒ clean 'no coverage'; catastrophic at all α ⇒ "
             "shared directions are off-manifold. Read results/steer_dir_diag/diag.json for the actual text."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / "s1_steer_dir_diag_report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(log_cost("S1", "steer_dir_diag", time.time() - t0, device, notes="steering direction diagnostic"))


if __name__ == "__main__":
    main()
