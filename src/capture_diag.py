"""CAPTURE DIAGNOSTIC — is the routing info-limit representational or fundamental? Fix a menu (the selected
basis + null); train the SAME bandit-as-ranker on different FEATURE SETS and measure how much of the oracle
headroom the router realizes at best-of-2:

  captured = (realized_router − random_floor) / (oracle_ceiling − random_floor)

Feature sets (whichever files exist):
  * generic       — distilroberta(prompt)          (enc_embed.npz; the current router encoder)
  * rm_prompt     — RM(prompt)                      (reward-aware, prompt only)
  * rm_prompt_base— RM(prompt, base generation)     (reward-aware, prompt + the model's no-move output)

If capture rises generic -> rm_prompt -> rm_prompt_base, the limit is REPRESENTATIONAL and (prompt, base)
is the routable signal — the green light for reward-encoder routing and the self-refine/revision-move
direction. If it stays flat, the limit is fundamental and the small-selected-basis story stands.

    python src/capture_diag.py --K 8 --seeds 12
"""

import argparse
import json
from pathlib import Path

import numpy as np

from menu_size_sweep import _bo, _rank_oracle, _rank_random
from bandit_ranker_val import train_policy
from router_bandit import _pca, _boot, _R_from_M
from prompt_basis import _greedy_submodular

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "prompt_basis_candpool_7b"
HP = dict(hidden=0, lr=0.02, beta=0.1, batch=64, epochs=60)


def _feature_sets():
    fs = {}
    enc = OUT / "enc_embed.npz"
    if enc.exists():
        fs["generic"] = np.load(enc, allow_pickle=True)["Htr"]
    pb = OUT / "enc_embed_prompt_base.npz"
    if pb.exists():
        fs["generic_prompt_base"] = np.load(pb, allow_pickle=True)["Htr"]   # same encoder, +base gen
    for which in ("prompt", "prompt_base"):
        f = OUT / f"rm_feats_{which}.npz"
        if f.exists():
            fs[f"rm_{which}"] = np.load(f, allow_pickle=True)["F"]
    return fs


def run(M, feats, K, seeds, n_pca, budget=2):
    C = M.shape[1]
    Mn = np.concatenate([M, np.zeros((len(M), 1))], axis=1)          # null arm (base, swing 0)
    res = {name: {"realized": [], "captured": []} for name in feats}
    rand_all, orac_all = [], []
    for seed in range(seeds):
        idx = np.arange(len(M)); np.random.default_rng(seed).shuffle(idx)
        a = int(0.75 * len(idx)); tr, ev = idx[:a], idx[a:]
        order_tr, _ = _greedy_submodular(M[tr], K)                    # menu selected on TRAIN only
        cols = list(order_tr[:K]) + [C]                              # + null
        Mtr, Mev = Mn[np.ix_(tr, cols)], Mn[np.ix_(ev, cols)]
        rng = np.random.default_rng(seed + 5)
        rand = _bo(_rank_random(len(cols), len(ev), rng), Mev, budget)
        orac = _bo(_rank_oracle(Mev), Mev, budget)
        rand_all.append(rand); orac_all.append(orac)
        for name, H in feats.items():
            Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], min(n_pca, H.shape[1]))
            rk = train_policy(Ztr, _R_from_M(Mtr), HP, seed)(Zev)
            realized = _bo(rk, Mev, budget)
            res[name]["realized"].append(realized)
            res[name]["captured"].append((realized - rand) / (orac - rand) if orac > rand else 0.0)
    return res, np.array(rand_all), np.array(orac_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n_pca", type=int, default=40)
    args = ap.parse_args()
    M = np.load(OUT / "swing_train.npz", allow_pickle=True)["M"]
    feats = _feature_sets()
    if not feats:
        raise SystemExit("no feature sets found (need at least enc_embed.npz)")
    res, rand, orac = run(M, feats, args.K, args.seeds, args.n_pca)

    order = [n for n in ("generic", "generic_prompt_base", "rm_prompt", "rm_prompt_base") if n in res]
    rows = ["# Capture diagnostic — is the routing info-limit representational or fundamental?\n",
            f"Menu = greedy top-{args.K} moves + null (base), budget=best-of-2, {args.seeds} seeds, 75/25 split, "
            f"bandit-as-ranker. captured = (router − random) / (oracle − random) of the best-of-2 headroom.\n",
            f"Random floor {rand.mean():+.3f} · Oracle ceiling {orac.mean():+.3f} "
            f"(headroom {orac.mean()-rand.mean():+.3f}).\n",
            "| features | realized best-of-2 | headroom captured |", "|---|---|---|"]
    for name in order:
        r = np.array(res[name]["realized"]); c = np.array(res[name]["captured"])
        rows.append(f"| {name} | {r.mean():+.3f} | {100*c.mean():.0f}% |")
    rows.append("")
    if "generic_prompt_base" in res and "generic" in res:
        d = np.array(res["generic_prompt_base"]["realized"]) - np.array(res["generic"]["realized"])
        lo, hi = _boot(d)
        verdict = ("⇒ the base generation carries ROUTABLE signal — base-conditioning helps (significant)."
                   if lo > 0 else
                   "⇒ base-conditioning HURTS." if hi < 0 else
                   f"⇒ PROMISING but underpowered: base-conditioning {'lifts' if d.mean() > 0 else 'shifts'} "
                   f"capture (see table) with CI through 0 — needs more seeds/prompts to confirm.")
        rows.append(f"- generic(prompt+base) − generic(prompt): **{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]** "
                    "(SAME encoder, +base gen — the clean (2) test, no RM-pooling confound) " + verdict)
    if "rm_prompt" in res and "generic" in res:
        d = np.array(res["rm_prompt"]["realized"]) - np.array(res["generic"]["realized"])
        lo, hi = _boot(d)
        rows.append(f"- RM(prompt) − generic(prompt): **{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]** "
                    + ("⇒ a reward-aware encoding of the PROMPT ALONE already routes better (partly representational)."
                       if lo > 0 else "⇒ prompt-only reward features don't beat generic (prompt alone is the limit)."))
    if "rm_prompt_base" in res and "generic" in res:
        d = np.array(res["rm_prompt_base"]["realized"]) - np.array(res["generic"]["realized"])
        lo, hi = _boot(d)
        rows.append(f"- RM(prompt, base) − generic(prompt): **{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]** "
                    + ("⇒ conditioning on the base generation UNLOCKS routable signal — green light for "
                       "reward-encoder routing + revision-move (self-refine) direction."
                       if lo > 0 else "⇒ even prompt+base doesn't help — the info-limit is fundamental; "
                       "the small-selected-basis story stands."))
    if "rm_prompt_base" not in res:
        rows.append("- _rm_prompt_base pending: needs base gens WITH TEXT (run base_text gen job, then "
                    "extract_rm_feats --which prompt_base)._")
    rpt = REPO_ROOT / "basis" / "s1_capture_diag_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
