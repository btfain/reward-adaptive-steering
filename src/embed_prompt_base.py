"""Clean test of the (2) hypothesis, decoupled from the RM-pooling confound: encode (prompt + base
generation) with the SAME good encoder as generic(prompt) and see if the base gen adds routable signal.
generic(prompt) vs generic(prompt+base) isolates base-conditioning from 'RM-as-encoder'. Local, free.

    python src/embed_prompt_base.py   ->  results/prompt_basis_candpool_7b/enc_embed_prompt_base.npz
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bakeoff_rankers import embed

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "prompt_basis_candpool_7b"
ENCODER = "distilroberta-base"
SPLIT, N = "train_large", 96


def main():
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[SPLIT][:N]
    by_pi = defaultdict(list)
    for l in open(REPO_ROOT / "results" / "end_banner_7b" / "base_text_train.jsonl"):
        r = json.loads(l)
        if r.get("text", "").strip():
            by_pi[r["pi"]].append((r["rm"], r["text"]))
    texts = []
    for pi, p in enumerate(prompts):
        cand = sorted(by_pi.get(pi, []))
        base = cand[len(cand) // 2][1] if cand else ""          # median-RM base gen (matches extract_rm_feats)
        texts.append(f"{p}\n\n[Draft response]\n{base}")
    H = embed(ENCODER, texts, max_len=512)                    # 512 = distilroberta max; 160 would truncate the base
    out = OUT / "enc_embed_prompt_base.npz"
    np.savez(out, Htr=H)
    print(f"embedded {H.shape} (prompt+base, {ENCODER}) -> {out}")


if __name__ == "__main__":
    main()
