"""P1(i) — candidate discovery: the LLM-from-signal move generator (replaces the hand-curated file).
Reads the cached base pool (completions + RM scores), forms contrastive high- vs low-reward examples,
and prompts a generator LLM to propose GENERAL, REUSABLE procedural moves (system-prompt instructions);
then verifies (length/parse) + semantic-dedups (distilroberta cosine). Reward-driven signal, from data
we already have. Output = a candidate file; combine with the curated file and compute the swing matrix
over the union (prompt_basis, sharded) to evaluate auto-vs-curated basis value (and seed P1(ii)).

    python src/gen_candidates.py --base-config configs/base_7b.yaml --pool results/prompt_basis_large_7b/pool.jsonl \
        --n_calls 12 --target 40 --out configs/candidates_auto_7b.txt
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from models import REPO_ROOT, generate, load_base, load_config, resolve_device

META_SYS = ("You design reusable, general-purpose procedural instructions (system prompts) that make an "
            "AI assistant produce higher-quality answers. Each instruction must be a single imperative "
            "sentence, general (not tied to any specific question), and about HOW to answer.")


def _meta_user(examples, n_ask):
    blocks = []
    for q, hi, lo in examples:
        blocks.append(f"QUESTION: {q[:200]}\nHIGHER-SCORING answer (excerpt): {hi[:300]}\n"
                      f"LOWER-SCORING answer (excerpt): {lo[:300]}")
    return ("Below are questions with a higher- and a lower-scoring answer under a quality model.\n\n"
            + "\n\n".join(blocks)
            + f"\n\nInfer WHAT the higher-scoring answers do differently, then propose {n_ask} DISTINCT, "
            "general, reusable one-sentence instructions that would push an assistant toward the "
            "higher-scoring style on ANY question. CRITICAL: each instruction must be DOMAIN-AGNOSTIC — do "
            "NOT mention any specific topic, task type (e.g. translation), brand, name, or detail from the "
            "examples above; it must apply to any question. Write in English. Output ONLY the instructions, "
            "one per line, no numbering.")


def _parse(text):
    out = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"')
        ascii_frac = sum(c.isascii() for c in s) / max(len(s), 1)
        if 15 <= len(s) <= 300 and s[0].isalpha() and ascii_frac > 0.95:   # verify: English-only, no CJK
            out.append(s)
    return out


def _embed_st(cands, model="sentence-transformers/all-MiniLM-L6-v2"):
    """Sentence-transformer embeddings (NOT a plain LM — distilroberta gives cosine≈0.98 for everything,
    which collapsed the pool). MiniLM: distinct sentences ~0.0-0.5, true duplicates ~0.9+."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model); mdl = AutoModel.from_pretrained(model).eval()
    with torch.no_grad():
        e = tok(cands, padding=True, truncation=True, max_length=64, return_tensors="pt")
        h = mdl(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
        E = ((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy()
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def _kmeans(X, K, iters=100, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        newC = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(K)])
        newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-9)
        if np.allclose(newC, C, atol=1e-5):
            C = newC; break
        C = newC
    return C


def _cluster_down(cands, target, near_dup=0.9):
    """Generate-many-then-cluster-down: drop true near-duplicates, then k-means to `target` and keep the
    medoid (closest to centroid) per cluster — diverse by construction. Returns (reps, global indices)."""
    E = _embed_st(cands)
    ki = []
    for i in range(len(cands)):                                  # remove true near-duplicates first
        if not ki or (E[i] @ E[ki].T).max() < near_dup:
            ki.append(i)
    if len(ki) <= target:
        return [cands[i] for i in ki], ki
    Ek = E[ki]; C = _kmeans(Ek, target)
    assign = np.argmax(Ek @ C.T, axis=1)
    reps, idx = [], []
    for j in range(target):
        mem = np.where(assign == j)[0]
        if len(mem):
            gi = ki[int(mem[int(np.argmax(Ek[mem] @ C[j]))])]
            reps.append(cands[gi]); idx.append(gi)
    return reps, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--generator", default=None, help="override base_model for generation")
    ap.add_argument("--pool", default="results/prompt_basis_large_7b/pool.jsonl")
    ap.add_argument("--n_calls", type=int, default=250)         # ~2k raw candidates; cluster AFTER smoke
    ap.add_argument("--examples_per", type=int, default=3)
    ap.add_argument("--ask_per", type=int, default=10)
    ap.add_argument("--out", default="configs/candidates_raw_7b.txt")
    ap.add_argument("--provenance", default="configs/candidates_raw_7b.provenance.json",
                    help="candidate -> source prompts it was derived from (for the smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base_cfg = load_config(args.base_config)
    if args.generator:
        base_cfg = dict(base_cfg); base_cfg["base_model"] = args.generator
    device = resolve_device(base_cfg)

    # contrastive high/low-reward base completions per prompt
    by = defaultdict(list)
    for l in open(REPO_ROOT / args.pool):
        r = json.loads(l)
        if r["completion"].strip():
            by[(r["split"], r["pi"])].append((r["prompt"], r["completion"], r["rm"]))
    contrasts = []
    for k, v in by.items():
        if len(v) >= 2:
            v = sorted(v, key=lambda x: x[2])
            if v[-1][2] - v[0][2] > 0.3:                    # need a real reward gap
                contrasts.append((v[-1][0], v[-1][1], v[0][1]))
    rng = np.random.default_rng(args.seed); rng.shuffle(contrasts)

    model, tok = load_base(base_cfg, device)
    gcfg = {"steer_layer": base_cfg["steer_layer"], "generation": {
        "max_new_tokens": 400, "do_sample": True, "temperature": 0.9, "top_p": 0.95}}
    raw, raw_src = [], []                                       # track SOURCE prompts per candidate (provenance)
    for i in range(args.n_calls):
        ex = contrasts[(i * args.examples_per) % len(contrasts):][:args.examples_per]
        if len(ex) < args.examples_per:
            ex = contrasts[:args.examples_per]
        src = [e[0] for e in ex]                                # the source questions for this call
        txt = generate(model, tok, _meta_user(ex, args.ask_per), gcfg, system=META_SYS)
        got = _parse(txt); raw += got; raw_src += [src] * len(got)
        print(f"  call {i+1}/{args.n_calls}: +{len(got)} (total {len(raw)})", flush=True)

    # exact-dedup only — clustering now happens AFTER the smoke test (on survivors). Keep provenance.
    seen, uniq, uniq_src = set(), [], []
    for c, s in zip(raw, raw_src):
        key = c.lower().rstrip(".")
        if key not in seen:
            seen.add(key); uniq.append(c); uniq_src.append(s)
    outp = REPO_ROOT / args.out
    outp.write_text("# RAW auto candidate moves (large pool; smoke-filter then cluster). gen_candidates.py\n"
                    + "\n".join(uniq) + "\n")
    json.dump({c: s for c, s in zip(uniq, uniq_src)}, open(REPO_ROOT / args.provenance, "w"))
    print(f"\n{len(raw)} raw -> {len(uniq)} exact-unique -> {outp}")
    print(f"provenance (candidate -> {args.examples_per} source prompts) -> {args.provenance}")
    print("\nsample:\n" + "\n".join(f"  - {c}" for c in uniq[:8]))


if __name__ == "__main__":
    main()
