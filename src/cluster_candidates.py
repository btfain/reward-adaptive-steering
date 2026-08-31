"""P1(i) — light clustering of the SMOKE-VERIFIED candidate pool + three cluster-representation methods.
Cluster the survivors LIGHTLY (reduce by --factor, ~4-8) so this mostly de-duplicates rather than
compressing viable options (some redundancy is fine — the submodular selection handles it). Save the FULL
clusters, then represent each cluster three ways for downstream comparison:
  (i)   medoid   — the member closest to the cluster centroid (most central real move)
  (ii)  random   — a random member
  (iii) summary  — the generator LLM summarizes the cluster's members into one instruction
Each representation is written as its own pool + a curated-union (n_auto marker) for the reference matrix.

    python src/cluster_candidates.py --base-config configs/base_7b.yaml --factor 6
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gen_candidates import _embed_st, _kmeans
from models import REPO_ROOT, generate, load_base, load_config, resolve_device

SUM_SYS = ("You merge several similar procedural instructions into ONE clear, general, reusable "
           "one-sentence instruction that captures their shared intent. Output only the single instruction.")


def _read(path):
    return [l.strip() for l in open(REPO_ROOT / path) if l.strip() and not l.startswith("#")]


def _write_pool(reps, curated, path, combined_path, label):
    (REPO_ROOT / path).write_text(f"# cluster representation = {label}\n" + "\n".join(reps) + "\n")
    seen = {c.lower().rstrip(".") for c in reps}
    cur_new = [c for c in curated if c.lower().rstrip(".") not in seen]
    combo = reps + cur_new
    (REPO_ROOT / combined_path).write_text(
        f"# combined pool ({label} auto + curated). n_auto={len(reps)}\n" + "\n".join(combo) + "\n")
    print(f"  {label}: {len(reps)} auto + {len(cur_new)} curated = {len(combo)} -> {combined_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--verified", default="configs/candidates_verified_7b.txt")
    ap.add_argument("--curated", default="configs/candidates_seed_v2.txt")
    ap.add_argument("--factor", type=int, default=6, help="reduce survivors by this factor (light: 4-8)")
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cands = _read(args.verified); curated = _read(args.curated)
    K = max(1, len(cands) // args.factor)
    E = _embed_st(cands)
    C = _kmeans(E, K, seed=args.seed)
    assign = np.argmax(E @ C.T, axis=1)
    rng = np.random.default_rng(args.seed)

    clusters = {int(j): [cands[i] for i in np.where(assign == j)[0]] for j in range(K) if (assign == j).any()}
    json.dump(clusters, open(REPO_ROOT / f"configs/candidate_clusters_{args.tag}.json", "w"), indent=1)
    print(f"{len(cands)} verified -> {len(clusters)} clusters (factor {args.factor}); full clusters saved.")

    medoid, random_rep = [], []
    for j, members in clusters.items():
        idx = np.where(assign == j)[0]
        medoid.append(cands[int(idx[int(np.argmax(E[idx] @ C[j]))])])   # closest to centroid
        random_rep.append(members[int(rng.integers(len(members)))])

    # (iii) LLM summary per cluster (GPU)
    base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
    model, tok = load_base(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 60, "do_sample": False, "temperature": 1.0, "top_p": 1.0}}
    summary = []
    for n, (j, members) in enumerate(clusters.items()):
        if len(members) == 1:
            summary.append(members[0]); continue
        txt = generate(model, tok, "Merge these into a single one-sentence instruction:\n"
                       + "\n".join(f"- {m}" for m in members[:12]), gcfg, system=SUM_SYS)
        s = txt.strip().splitlines()[0].strip().strip('"-•* ') if txt.strip() else members[0]
        summary.append(s if 15 <= len(s) <= 300 else members[0])
        if (n + 1) % 25 == 0:
            print(f"  summary {n+1}/{len(clusters)}", flush=True)

    print("representations:")
    _write_pool(medoid, curated, f"configs/candidates_medoid_{args.tag}.txt",
                f"configs/candidates_medoid_combined_{args.tag}.txt", "medoid")
    _write_pool(random_rep, curated, f"configs/candidates_random_{args.tag}.txt",
                f"configs/candidates_random_combined_{args.tag}.txt", "random")
    _write_pool(summary, curated, f"configs/candidates_summary_{args.tag}.txt",
                f"configs/candidates_summary_combined_{args.tag}.txt", "summary")
    print(f"\nfull clusters -> configs/candidate_clusters_{args.tag}.json  (3 representations written)")


if __name__ == "__main__":
    main()
