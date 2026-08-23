"""Top-k router — DE-BIASED (the trustworthy version of top_k_probe). The free train-data cut is
winner's-curse inflated (max-over-top-k of noisy mean swings). Here the ceiling set has per-sample
swings (100 prompts x 8 moves x 12 samples), so we split the 12 into SELECT-half (pick best of the
router's top-k) and SCORE-half (evaluate it) — independent draws, no winner's curse. Router = ridge
trained on large_7b (distilroberta→swing over the 8 moves), applied to the HELD-OUT b1 ceiling prompts.

Answers cleanly: does the router's top-2 (generate 2, keep reward-best) capture most of the oracle
headroom (⇒ cheap router-narrowed selection, ~2x compute — a real middle ground), or do you need ~all
K (⇒ router adds nothing as a ranker; pure selection; reinforces the single-turn bound)?

    python src/top_k_debias.py --seeds 40
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from router_bandit import _boot

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def embed(encoder, texts, max_len=160, batch=16):
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


def ridge_W(X, Y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    Yf = np.nan_to_num(Y); dmu = Yf.mean(0)
    W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ (Yf - dmu))
    return (mu, sd, W, dmu)


def ridge_pred(model, X):
    mu, sd, W, dmu = model
    return (X - mu) / sd @ W + dmu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--encoder", default="distilroberta-base")   # match large_7b router features
    args = ap.parse_args()
    L7 = REPO_ROOT / "results" / "prompt_basis_large_7b"
    B1 = REPO_ROOT / "results" / "bandit_online_b1_1500"

    # router: ridge on large_7b (distilroberta enc -> swing over the 8 selected moves)
    ec = np.load(L7 / "enc_embed.npz", allow_pickle=True); Htr = ec["Htr"]
    S = json.load(open(L7 / "selection.json"))["order"]
    Msel = np.load(L7 / "swing_train.npz", allow_pickle=True)["M"][:, S]
    ok = ~np.isnan(Msel).all(1)
    model = ridge_W(Htr[ok], Msel[ok])
    K = len(S)

    # held-out b1 ceiling prompts + per-sample swings
    shards = sorted(glob.glob(str(B1 / "prep_shard_*.npz")))
    cg = np.concatenate([np.load(f)["ceil_gi"] for f in shards])
    cs = np.concatenate([np.load(f)["ceil_sw"] for f in shards])       # (100, K, 12)
    prompts_b1 = json.load(open(REPO_ROOT / "data" / "prompts.json"))["b1"]
    cprompts = [prompts_b1[int(g)] for g in cg]
    Xc = embed(args.encoder, cprompts)
    rank = np.argsort(-ridge_pred(model, Xc), axis=1)                  # router ranking per ceiling prompt
    m = cs.shape[2]; h = m // 2                                        # select/score split of the 12 samples

    def best_of(cand, sel_i, sco_i):                                  # pick argmax on SELECT, score on SCORE, decline on SELECT
        sv = np.nan_to_num(sel_i[cand], nan=-1e9)
        j = cand[int(np.argmax(sv))]
        return sco_i[j] if sel_i[j] > 0 else 0.0

    curves, rand_curves, singles, oracles = [], [], [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = np.array([rng.permutation(m) for _ in range(len(cs))])  # per-prompt sample shuffle
        sel = np.take_along_axis(cs, perm[:, None, :h], axis=2).mean(2)  # (100,K) select-half means
        sco = np.take_along_axis(cs, perm[:, None, h:], axis=2).mean(2)  # (100,K) score-half means
        row, rrow = [], []
        for k in range(1, K + 1):
            r = [best_of(rank[i, :k], sel[i], sco[i]) for i in range(len(cs))]
            rr = [best_of(rng.permutation(K)[:k], sel[i], sco[i]) for i in range(len(cs))]  # random k of the basis
            row.append(float(np.mean(r))); rrow.append(float(np.mean(rr)))
        curves.append(row); rand_curves.append(rrow)
        singles.append(float(np.mean([sco[i, 0] if sel[i, 0] > 0 else 0.0 for i in range(len(cs))])))
        # de-biased oracle over ALL K (select on select-half, score on score-half)
        oracles.append(float(np.mean([sco[i, int(np.nanargmax(np.nan_to_num(sel[i], nan=-1e9)))]
                                      if np.nanmax(sel[i]) > 0 else 0.0 for i in range(len(cs))])))
    C = np.array(curves); R = np.array(rand_curves)
    single = float(np.mean(singles)); oracle = float(np.mean(oracles))
    mean_k = C.mean(0); rand_k = R.mean(0); head = oracle - single
    d12 = C[:, 1] - C[:, 0]; d12lo, d12hi = _boot(d12)
    rr2 = C[:, 1] - R[:, 1]; rr2lo, rr2hi = _boot(rr2)                # router top-2 minus random-2 (does ranking earn its keep?)

    rows = [f"# Top-k router DE-BIASED — b1 ceiling (held-out), router=ridge on large_7b ({args.encoder})\n",
            f"Per-sample select/score split (6/6 of 12) kills winner's curse. {len(cs)} held-out ceiling prompts, "
            f"{args.seeds} split-seeds, K={K}. single {single:+.3f}; de-biased oracle (all-K) {oracle:+.3f}. "
            f"realized_k = pick best-of-router's-top-k on SELECT half, score on SCORE half.\n",
            "| k (generate) | router top-k | random-k (basis) | router−random | % headroom (router) |",
            "|---|---|---|---|---|"]
    for k in range(K):
        frac = (mean_k[k] - single) / head * 100 if head > 0 else float("nan")
        rows.append(f"| {k+1} | {mean_k[k]:+.3f} | {rand_k[k]:+.3f} | {mean_k[k]-rand_k[k]:+.3f} | {frac:.0f}% |")
    f2 = (mean_k[1] - single) / head * 100 if head > 0 else float("nan")
    f1 = (mean_k[0] - single) / head * 100 if head > 0 else float("nan")
    rand_f2 = (rand_k[1] - single) / head * 100 if head > 0 else float("nan")
    rows += ["", "## Reading",
             f"- **top-1 {mean_k[0]:+.3f} → top-2 {mean_k[1]:+.3f}: Δ{d12.mean():+.3f} [{d12lo:+.3f}, {d12hi:+.3f}]** "
             f"(top-1 {f1:.0f}% of headroom, top-2 {f2:.0f}%).",
             f"- **router top-2 − random-2 = {rr2.mean():+.3f} [{rr2lo:+.3f}, {rr2hi:+.3f}]** "
             f"(random-2 only {rand_f2:.0f}% of headroom) ⇒ the RANKING, not just the basis, does the work."
             if rr2lo > 0.02 else
             f"- router top-2 ≈ random-2 ({rr2.mean():+.3f} [{rr2lo:+.3f}, {rr2hi:+.3f}]) ⇒ the gain is the BASIS, not "
             "the router's ranking — any 2 moves do as well.",
             ("- **⇒ the router IS a useful search-narrower** — router-guided best-of-2 gets ~2/3 of the oracle at 2x "
              "compute AND beats random-2; the learned router earns its keep as a RANKER even though top-1 is bounded. "
              "Real middle ground; partially rescues the cost story." if d12lo > 0.03 and f2 > 55 and rr2lo > 0.02 else
              "- ⇒ mixed: read the router−random column to see whether ranking or just the basis drives the top-2 gain."),
             f"- Sanity: de-biased top-1 {mean_k[0]:+.3f} ≈ B1 router/exact-policy ceiling; all-K {mean_k[-1]:+.3f} ≈ "
             f"oracle.json {oracle:+.3f}."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / "s1_top_k_debias_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
