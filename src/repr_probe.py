"""B0.5 — FREE representation probe. Does a bigger / stronger FROZEN encoder raise the extractable
ceiling above distilroberta's +0.37? Embeds the existing 450 train prompts with several frozen
encoders (mean-pooled), then re-runs the offline exact-policy + regression ceiling on the SAME
cached swings (swing_train.npz) via router_bandit's fit machinery. No generation, no fine-tuning —
a lower bound on what each representation contains off-the-shelf.

Positive result (ceiling rises) => representation is the lever => B1 GPU scaling is justified, and
fine-tuning that encoder becomes a first-class B1 arm (fine-tuning needs the scaled data; it overfit
at 450 prompts in router_encoder.py). Flat => scaling that frozen rep won't help; the fine-tune-at-
scale question moves into B1 proper.

    python src/repr_probe.py --tag large_7b --encoders distilroberta-base,sentence-transformers/all-mpnet-base-v2,intfloat/e5-large-v2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from router_bandit import _pca, _boot, _realized, _R_from_M, fit_policy_exact, fit_regression, _select

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def _feat(Hfit, Hs, n_pca):
    """n_pca>0 & <dim: PCA (small-data regularizer). Else: standardized RAW embeddings (full dim,
    let dropout/weight-decay/early-stopping regularize) — keeps low-variance task directions PCA drops."""
    if 0 < n_pca < Hfit.shape[1]:
        return _pca(Hfit, Hs, n_pca)
    mu, sd = Hfit.mean(0), Hfit.std(0) + 1e-6
    return [(H - mu) / sd for H in Hs]


def embed(encoder, prompts, batch=16):
    tok = AutoTokenizer.from_pretrained(encoder)
    mdl = AutoModel.from_pretrained(encoder).eval()
    prefix = "query: " if "e5" in encoder.lower() else ""            # e5 expects a prefix; others none
    texts = [prefix + p for p in prompts]
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            e = tok(texts[s:s + batch], padding=True, truncation=True, max_length=160, return_tensors="pt")
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--config", default="configs/prompt_basis_large_7b.yaml")
    ap.add_argument("--encoders", default="distilroberta-base,sentence-transformers/all-mpnet-base-v2,intfloat/e5-large-v2")
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--curve", action="store_true", help="learning curve: exact-policy ceiling vs train-set size (data- vs info-limited)")
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"

    import yaml
    pb = yaml.safe_load(open(REPO_ROOT / args.config))
    allp = json.load(open(REPO_ROOT / "data" / "prompts.json"))[pb.get("prompts_split", "train")]
    prompts = allp[:pb["pool"]["n_prompts_train"]]
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]; K = len(S)
    idx_all = np.where(~np.isnan(Msel).all(1))[0]

    # full-dim raw needs stronger regularization (and the user's deeper head); PCA stays lean
    raw = not (0 < args.n_pca)
    hlist = (0, 128, 256) if raw else (0, 64)
    wds = (0.01, 0.1, 0.3) if raw else (0.01, 0.1)
    drops = (0.3, 0.5) if raw else (0.3,)
    grid_reg = [dict(hidden=h, dropout=d, lr=0.05, wd=wd, epochs=800, patience=40)
                for h in hlist for wd in wds for d in drops]
    grid_exact = [dict(hidden=h, dropout=d, lr=0.05, wd=wd, beta=b, epochs=800, patience=40)
                  for h in hlist for wd in wds for d in drops for b in (0.0, 0.01)]

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    if args.curve:                                                   # data-limited vs information-limited
        fracs = [0.25, 0.5, 0.75, 1.0]
        hp = dict(hidden=128, dropout=0.5, lr=0.05, wd=0.1, beta=0.0, epochs=800, patience=40)
        rows = [f"# B0.5 learning curve — exact-policy ceiling vs train size (full-dim, {args.seeds} seeds)\n",
                f"Is the ~+0.38 frozen ceiling data-limited (rises with prompts ⇒ B1 scaling helps) or information-"
                f"limited (flat ⇒ signal not in the text ⇒ pivot)? single +0.300, oracle ≈ +0.78.\n",
                "| encoder | " + " | ".join(f"n_tr≈{int(f*270)}" for f in fracs) + " |",
                "|---|" + "---|" * len(fracs)]
        trends = {}
        for enc in encoders:
            try:
                H = embed(enc, prompts)
            except Exception as e:
                rows.append(f"| {enc} | SKIP {str(e)[:40]} |"); continue
            cells, means = [], []
            for f in fracs:
                ev = []
                for seed in range(args.seeds):
                    idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
                    n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
                    tr, va, evi = idx[:a], idx[a:b], idx[b:]
                    sub = tr[:max(8, int(f * len(tr)))]
                    Ztr, Zva, Zev = _feat(H[sub], [H[sub], H[va], H[evi]], 0)
                    act = fit_policy_exact(Ztr, _R_from_M(Msel[sub]), Zva, _R_from_M(Msel[va]), hp, "cpu", seed)
                    ev.append(_realized(act(Zev), _R_from_M(Msel[evi])).mean())
                cells.append(f"{np.mean(ev):+.3f}"); means.append(np.mean(ev))
            trends[enc] = means[-1] - means[0]
            rows.append(f"| {enc} | " + " | ".join(cells) + " |")
        rows += ["", "## Reading",
                 f"- Δ(full−quarter) per encoder: " + ", ".join(f"{e} {d:+.3f}" for e, d in trends.items()) + ".",
                 "- **Still climbing at n_tr=270** ⇒ data-limited ⇒ B1 (more prompts) can lift the ceiling — scale.",
                 "- **Flat by n_tr=270** ⇒ information-limited ⇒ the prompt text lacks the best-move signal ⇒ more prompts "
                 "won't help; carry the working bandit to Study 2 (multi-turn state carries richer signal)."]
        BASIS.mkdir(exist_ok=True)
        rpt = BASIS / f"s1_repr_curve_{args.tag}_report.md"
        rpt.write_text("\n".join(rows) + "\n")
        print("\n".join(x for x in rows if not x.startswith("|")))
        print(f"report -> {rpt}")
        return

    results, dims, single_all = {}, {}, []
    for enc in encoders:
        t0 = time.time()
        try:
            H = embed(enc, prompts)
        except Exception as e:                                       # missing dep / download failure -> skip, don't crash
            results[enc] = ("SKIP", str(e)[:80]); continue
        dims[enc] = H.shape[1]
        ex_evals, rg_evals, singles = [], [], []
        for seed in range(args.seeds):
            idx = idx_all.copy(); np.random.default_rng(seed).shuffle(idx)
            n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
            tr, va, ev = idx[:a], idx[a:b], idx[b:]
            Ztr, Zva, Zev = _feat(H[tr], [H[tr], H[va], H[ev]], args.n_pca)
            Rtr, Rva, Rev = _R_from_M(Msel[tr]), _R_from_M(Msel[va]), _R_from_M(Msel[ev])
            singles.append(np.nan_to_num(Msel[ev, 0], nan=0.0).mean())
            _, ex = _select(fit_policy_exact, grid_exact, Ztr, Rtr, Zva, Rva, Rva, (Zev, Rev), "cpu", seed)
            _, rg = _select(fit_regression, grid_reg, Ztr, Msel[tr], Zva, Msel[va], Rva, (Zev, Rev), "cpu", seed)
            ex_evals.append(ex.mean()); rg_evals.append(rg.mean())
        results[enc] = (np.array(ex_evals), np.array(rg_evals), np.array(singles), time.time() - t0)
        single_all = results[enc][2]

    single = single_all
    naive_or = np.mean([max(0.0, np.nan_to_num(Msel[i], nan=-1e9).max()) for i in idx_all])
    base_key = encoders[0]
    base_ex = results[base_key][0] if isinstance(results[base_key][0], np.ndarray) else None

    rows = [f"# B0.5 representation probe (frozen encoders, offline) — {args.tag}\n",
            f"Frozen mean-pooled embeddings of {len(prompts)} train prompts -> PCA-{args.n_pca} -> offline exact-policy + "
            f"regression ceiling on cached swings ({len(idx_all)} valid, K={K}), {args.seeds} seeds, honest val-selection. "
            f"NO generation, NO fine-tuning (lower bound). single {single.mean():+.3f}; de-biased oracle ≈ +0.78 (naive {naive_or:+.3f}).\n",
            "| encoder | dim | exact-policy eval | vs single | Δ vs distilroberta (paired) | regression eval |",
            "|---|---|---|---|---|---|"]
    for enc in encoders:
        r = results[enc]
        if isinstance(r[0], str):
            rows.append(f"| {enc} | — | SKIP ({r[1]}) | | | |"); continue
        ex, rg, sg, dt = r
        if base_ex is not None and isinstance(base_ex, np.ndarray) and enc != base_key:
            d = ex - base_ex; dlo, dhi = _boot(d)
            dcell = f"{d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        else:
            dcell = "— (baseline)"
        rows.append(f"| {enc} | {dims[enc]} | {ex.mean():+.3f} ± {ex.std():.3f} | {ex.mean()-single.mean():+.3f} | "
                    f"{dcell} | {rg.mean():+.3f} ± {rg.std():.3f} |")

    # verdict
    lifts = []
    for enc in encoders:
        r = results[enc]
        if isinstance(r[0], np.ndarray) and enc != base_key and base_ex is not None:
            d = r[0] - base_ex; dlo, _ = _boot(d)
            if dlo > 0:
                lifts.append((enc, d.mean()))
    rows += ["", "## Reading"]
    if lifts:
        best = max(lifts, key=lambda x: x[1])
        rows.append(f"- **{best[0]} lifts the exact-policy ceiling by {best[1]:+.3f} over distilroberta (paired CI > 0)** ⇒ "
                    "representation IS the lever ⇒ B1 GPU scaling justified; make fine-tuning this encoder a first-class B1 arm.")
    else:
        rows.append("- **No frozen encoder lifts the ceiling (all paired CIs straddle 0)** ⇒ off-the-shelf representation is not "
                    "the free win. This is a LOWER BOUND: fine-tuning at scale may still help, so that question moves into B1 "
                    "proper (frozen-head vs fine-tune arms) rather than being settled here.")
    rows.append(f"- Ceiling context: best frozen exact-policy vs de-biased oracle (~+0.78) shows how much conditioning remains "
                "unreachable from the prompt text alone.")
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_repr_probe_{args.tag}_pca{args.n_pca}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print("\n(table in report)")
    print(f"report -> {rpt}")


if __name__ == "__main__":
    main()
