"""Frozen mean-pooled embeddings of the prompt TEXT from a small encoder (default distilroberta-base)
-> results/prompt_basis_<tag>/enc_embed.npz {Htr, Hte}. Lets router_explore --rep enc route them with
the SAME PCA-40 + dropout/weight-decay grid + val-selection + early-stopping discipline used for the
LLM states — a FAIR, regularized test of the dedicated-encoder idea (the raw router_encoder head was
overfit). CPU is fine (~1 min for 600 short prompts).

    python src/embed_prompts.py --tag large_7b [--encoder distilroberta-base]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--config", default="configs/prompt_basis_large_7b.yaml")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pb = yaml.safe_load(open(REPO_ROOT / args.config))
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    allp = json.load(open(REPO_ROOT / "data" / "prompts.json"))[pb.get("prompts_split", "train")]
    ntr, nte = pb["pool"]["n_prompts_train"], pb["pool"]["n_prompts_test"]
    Ptr, Pte = allp[:ntr], allp[ntr:ntr + nte]

    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        out = []
        for s in range(0, len(texts), args.batch):
            e = tok(texts[s:s + args.batch], padding=True, truncation=True, max_length=160,
                    return_tensors="pt").to(device)
            h = enc(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
        return np.concatenate(out)

    Htr, Hte = embed(Ptr), embed(Pte)
    np.savez(OUT / "enc_embed.npz", Htr=Htr, Hte=Hte, encoder=np.array(args.encoder))
    print(f"frozen {args.encoder} embeddings ({Htr.shape[1]}-d) -> {OUT / 'enc_embed.npz'}  "
          f"(train {Htr.shape[0]}, test {Hte.shape[0]})")


if __name__ == "__main__":
    main()
