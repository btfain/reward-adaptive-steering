"""Add a larger 'train_large' split to data/prompts.json — fresh UltraFeedback prompts, deduped
against (and NOT disturbing) the existing train/heldout splits, so old pools stay valid. CPU/login-
node only (just streams text; no GPU, no model). Run once on the cluster, commit the updated file:

    .venv/bin/python src/expand_prompts.py --n 600
"""

import argparse
import json
import random

from datasets import load_dataset
from models import REPO_ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="size of the new split")
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--max-chars", type=int, default=400, help="prompt length filter (match basis.yaml)")
    ap.add_argument("--seed", type=int, default=7, help="!= the original split's seed, for fresh prompts")
    ap.add_argument("--split-name", default="train_large")
    args = ap.parse_args()

    d = load_config(args.base_config)["data"]
    path = REPO_ROOT / "data" / "prompts.json"
    data = json.load(open(path))
    existing = set()                                   # dedup against ALL existing splits (keep every one fresh)
    for v in data.values():
        existing |= set(v)

    stream = load_dataset(d["prompt_dataset"], split=d["prompt_split"], streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=4000)
    out, seen = [], set()
    for row in stream:
        p = row[d["prompt_field"]].strip()
        if 10 <= len(p) <= args.max_chars and p not in existing and p not in seen:
            seen.add(p); out.append(p)
        if len(out) == args.n:
            break
    if len(out) < args.n:
        raise RuntimeError(f"only found {len(out)} fresh prompts (< {args.n}); loosen --max-chars or --n")
    random.Random(args.seed).shuffle(out)

    data[args.split_name] = out
    json.dump(data, open(path, "w"), indent=1)
    print(f"added split '{args.split_name}' ({len(out)} prompts); kept train={len(data.get('train', []))}, "
          f"heldout={len(data.get('heldout', []))} -> {path}")


if __name__ == "__main__":
    main()
