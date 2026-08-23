"""Idea B (FREE, offline) — the 2-generation lever. B1 showed routing the best move FROM THE PROMPT is
bounded (fine-tune eval = single, big oracle gap). Our diagnosis: the "which move" signal needs per-prompt
TRIAL info not in the prompt text. The cheapest trial = ONE base generation (no prompt/steering). This
tests whether routing on (prompt + base_gen + RM(base_gen)) captures more of the oracle than prompt-only —
i.e. whether a 2-gen method (generate base, score it, route, generate move) beats 1-gen prompt routing.

All offline on cached large_7b data (pool.jsonl base gens + swing_train.npz), roberta-base frozen
embeddings, SAME PCA-40 + exact-policy + honest val-selection + multi-seed as router_bandit — the ONLY
thing that varies across arms is the input representation, so any lift is purely the trial info.

Arms: prompt (=B1 frozen baseline) | base_gen only | prompt+gen | prompt+gen+score.

    python src/idea_b_probe.py --tag large_7b --seeds 12
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from router_bandit import _pca, _boot, _R_from_M, _realized, fit_policy_exact, _select

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def embed(encoder, texts, max_len, batch=16):
    tok = AutoTokenizer.from_pretrained(encoder)
    mdl = AutoModel.from_pretrained(encoder).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            e = tok(texts[s:s + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"

    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]; K = len(S)
    n_prompts = Msel.shape[0]

    # one base gen + its RM per train prompt (mimics test-time: you generate ONE base sample, score it)
    by = defaultdict(list)
    for l in open(OUT / "pool.jsonl"):
        r = json.loads(l)
        if r["split"] == "train":
            by[r["pi"]].append((r["prompt"], r["completion"], r["rm"]))
    prompts, gens, scores = [], [], []
    for pi in range(n_prompts):
        recs = by.get(pi, [])
        rec = next((x for x in recs if x[1].strip()), recs[0] if recs else (None, "", 0.0))
        prompts.append(rec[0] or ""); gens.append(rec[1]); scores.append(rec[2])
    scores = np.array(scores, dtype=np.float32)
    zscore = ((scores - scores.mean()) / (scores.std() + 1e-6))[:, None]

    print("embedding prompt / base_gen (frozen roberta) ...")
    Ep = embed(args.encoder, prompts, max_len=160)           # prompt (short)
    Eg = embed(args.encoder, gens, max_len=384)              # base gen (long)

    feats = {"prompt": Ep,                                    # = B1 frozen baseline
             "base_gen": Eg,                                  # trial info alone
             "prompt+gen": np.concatenate([Ep, Eg], 1),       # both, no dilution
             "prompt+gen+score": np.concatenate([Ep, Eg], 1)} # score appended AFTER pca (below)

    idx_all = np.where(~np.isnan(Msel).all(1))[0]
    grid = [dict(hidden=h, dropout=0.3, lr=0.05, wd=wd, beta=b, epochs=800, patience=40)
            for h in (0, 64) for wd in (0.01, 0.1) for b in (0.0, 0.01)]

    acc = {k: [] for k in feats}; singles = []
    for seed in range(args.seeds):
        idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
        n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
        tr, va, ev = idx[:a], idx[a:b], idx[b:]
        Rtr, Rva, Rev = _R_from_M(Msel[tr]), _R_from_M(Msel[va]), _R_from_M(Msel[ev])
        singles.append(np.nan_to_num(Msel[ev, 0], nan=0.0).mean())
        for name, H in feats.items():
            Ztr, Zva, Zev = _pca(H[tr], [H[tr], H[va], H[ev]], args.n_pca)
            if name == "prompt+gen+score":                   # append the standardized base score as an explicit feature
                Ztr = np.concatenate([Ztr, zscore[tr]], 1)
                Zva = np.concatenate([Zva, zscore[va]], 1)
                Zev = np.concatenate([Zev, zscore[ev]], 1)
            _, ev_arr = _select(fit_policy_exact, grid, Ztr, Rtr, Zva, Rva, Rva, (Zev, Rev), "cpu", seed)
            acc[name].append(float(ev_arr.mean()))

    single = np.array(singles)
    base = np.array(acc["prompt"])
    oracle = np.mean([max(0.0, np.nan_to_num(Msel[i], nan=-1e9).max()) for i in idx_all])
    rows = [f"# Idea B — route on TRIAL info (prompt + base gen + score) — {args.tag} ({args.encoder}, offline)\n",
            f"Does one base generation supply the per-prompt signal the prompt lacked? Frozen roberta, PCA-{args.n_pca}, "
            f"exact-policy, {args.seeds} seeds, honest val-selection. single {single.mean():+.3f}; naive oracle "
            f"{oracle:+.3f} (de-biased ≈ +0.9). Only the INPUT varies.\n",
            "| routing input | eval ΔRM (mean±sd) | vs single | vs prompt-only (paired) |", "|---|---|---|---|"]
    for name in ("prompt", "base_gen", "prompt+gen", "prompt+gen+score"):
        v = np.array(acc[name])
        d = v - base; dlo, dhi = _boot(d)
        vs = "— (baseline)" if name == "prompt" else f"{d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        rows.append(f"| {name} | {v.mean():+.3f} ± {v.std():.3f} | {v.mean()-single.mean():+.3f} | {vs} |")

    best = max(("prompt+gen", "prompt+gen+score"), key=lambda k: np.mean(acc[k]))
    d = np.array(acc[best]) - base; dlo, dhi = _boot(d)
    rows += ["", "## Reading"]
    if dlo > 0:
        rows.append(f"- **{best} beats prompt-only by {d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}] (paired CI>0)** ⇒ the base "
                    "generation carries per-prompt trial signal the prompt lacked ⇒ a 2-gen method is the lever; "
                    "don't pivot — build online (generate base, score, route, generate move).")
    else:
        rows.append(f"- **{best} vs prompt-only {d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}] (CI includes 0)** ⇒ one base gen "
                    "does NOT add extractable signal ⇒ the trial info the oracle needs is more than a single sample "
                    "reveals ⇒ strengthens the single-turn bound; selection (Idea A) or multi-turn is the path.")
    rows.append(f"- Context: all arms vs the +{oracle:.2f} naive oracle — the gap that stays uncaptured is the "
                "conditioning that needs richer trial info than one base draw.")
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_idea_b_probe_{args.tag}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
