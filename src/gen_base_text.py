"""Generate BASE (no-move) generations WITH TEXT for the candpool train prompts, so we can extract the RM's
reward-aware encoding of (prompt, base generation) — the (2) half of the capture diagnostic. Our earlier gen
jobs cached only RM SCORES, not text; this saves both. Small: n_base samples per train prompt.

    python src/gen_base_text.py --base-config configs/base_7b.yaml --n 96 --n_base 4
    -> results/end_banner_7b/base_text_train.jsonl   {pi, kind:'base', rm, text}
"""

import argparse
import json
import time
from pathlib import Path

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--split", default="train_large")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--n_base", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.base_config); device = resolve_device(cfg); t0 = time.time()
    model, tok = load_base(cfg, device); rm, rm_tok = load_rm(cfg, device)
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[args.split][:args.n]
    g = cfg["generation"]
    gcfg = {"steer_layer": cfg["steer_layer"], "generation": {
        "max_new_tokens": g["max_new_tokens"], "do_sample": True,
        "temperature": g["temperature"], "top_p": g["top_p"]}}
    out = REPO_ROOT / "results" / "end_banner_7b"; out.mkdir(parents=True, exist_ok=True)
    fp = out / "base_text_train.jsonl"
    with open(fp, "w") as f:
        for pi, prompt in enumerate(prompts):
            for comp in generate_batch(model, tok, [prompt] * args.n_base, gcfg):
                if comp.strip():
                    f.write(json.dumps({"pi": pi, "kind": "base",
                                        "rm": rm_score(rm, rm_tok, prompt, comp), "text": comp}) + "\n")
            f.flush()
            if (pi + 1) % 10 == 0:
                print(f"  base-text: {pi+1}/{len(prompts)}", flush=True)
    print(log_cost("S1", "gen_base_text", time.time() - t0, device, notes="base gens w/ text for RM features"))
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
