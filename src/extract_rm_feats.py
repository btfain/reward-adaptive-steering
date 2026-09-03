"""Extract REWARD-MODEL representations to use as ROUTER features — the test of whether the info-limit on
'which move helps this prompt' is representational (a generic encoder just can't see it) or fundamental.

The RM (Skywork-Reward-V2, a decoder seq-classifier) scores a (prompt, response) pair from its LAST-token
hidden state through a value head. That last-token hidden state IS a reward-aware embedding. We tap it in
two forms:
  * prompt        — RM(prompt, response="")  : reward-aware encoding of the PROMPT alone (free, no gen).
  * prompt_base   — RM(prompt, base_gen)      : reward-aware encoding of the prompt AND the model's own
                    no-move generation — strictly more information (reveals the failure mode). Needs base
                    generations WITH TEXT (a small gen job saves them; we only cached scores before).

    python src/extract_rm_feats.py --which prompt
    python src/extract_rm_feats.py --which prompt_base --base-jsonl results/end_banner_7b/base_text_train.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models import load_config, load_rm, resolve_device

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "prompt_basis_candpool_7b"


@torch.no_grad()
def _feat(rm, tok, prompt, response):
    """Last-token last-layer hidden state of the RM for (prompt, response) — the reward-aware embedding."""
    conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    inp = tok.apply_chat_template(conv, return_tensors="pt", return_dict=True).to(rm.device)
    h = rm(**inp, output_hidden_states=True).hidden_states[-1]   # (1, T, d)
    return h[0, -1, :].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["prompt", "prompt_base"])
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--split", default="train_large")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--base-jsonl", default=None, help="base gens WITH TEXT (for --which prompt_base)")
    args = ap.parse_args()
    cfg = load_config(args.base_config); device = resolve_device(cfg)
    rm, tok = load_rm(cfg, device)
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[args.split][:args.n]

    resp = [""] * len(prompts)
    if args.which == "prompt_base":
        if not args.base_jsonl:
            raise SystemExit("--which prompt_base needs --base-jsonl (base gens with 'text')")
        # pick, per prompt, the base sample whose RM score is the MEDIAN (a typical base gen, not a lucky tail)
        by_pi = defaultdict(list)
        for l in open(REPO_ROOT / args.base_jsonl):
            r = json.loads(l)
            if r.get("kind", "base") == "base" and r.get("text", "").strip():
                by_pi[r["pi"]].append((r["rm"], r["text"]))
        for pi in range(len(prompts)):
            cand = sorted(by_pi.get(pi, []))
            resp[pi] = cand[len(cand) // 2][1] if cand else ""     # median-RM base gen (fallback: prompt-only)

    F = np.stack([_feat(rm, tok, p, r) for p, r in zip(prompts, resp)])
    out = OUT / f"rm_feats_{args.which}.npz"
    np.savez(out, F=F)
    print(f"extracted {F.shape} -> {out}  (d={F.shape[1]}, response={'base gen' if args.which=='prompt_base' else 'empty'})")


if __name__ == "__main__":
    main()
