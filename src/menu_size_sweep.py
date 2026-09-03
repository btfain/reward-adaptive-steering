"""P1 MENU-SIZE SWEEP — does selecting a small basis (stage ii) earn its keep, or should we keep the
larger clustered pool and let the online bandit route over all of it?

The tension: a LARGER menu has a higher oracle ceiling (more chance some move is great for a given prompt —
our data: auto>curated, gap grows with K), but is HARDER to route (random pick from a big menu is bad) and
HARDER for the bandit to explore (action space grows). A SMALL selected menu has a lower ceiling but even
weak routing lands on a decent move and the bandit converges fast. Which wins on REALIZED value?

This settles it OFFLINE on the already-computed candpool swing matrix (96 x 220) — ZERO new GPU. For a grid
of menu sizes K we train the SAME hardened bandit-as-ranker (the deployed router) on a train split and score
its realized best-of-2 on a held-out split, against the random-routing floor and the oracle ceiling, for two
menu families:
  * greedy   — submodular top-K (selection done honestly on the TRAIN split, per seed): "we selected"
  * random   — a random K-subset of the pool: "we kept the pool without selecting"
K=|pool| is the full "keep everything" arm. We also test whether TRAINING THE BANDIT HARDER (more epochs)
rescues the full menu — the user's proposed alternative to selection.

Caveat: the bandit reads the CACHED swing means as its reward oracle (no fresh-sample noise), which slightly
flatters every ranker equally — fair for the RELATIVE shape across K (all we need to decide), and the live
gen run remains the arbiter of absolute numbers.

    PYTHONPATH=src python src/menu_size_sweep.py --seeds 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bandit_ranker_val import train_policy
from router_bandit import _pca, _boot, _R_from_M
from bakeoff_rankers import embed
from prompt_basis import _greedy_submodular

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "prompt_basis_candpool_7b"
HP = dict(hidden=0, lr=0.02, beta=0.1, batch=64, epochs=60)   # the validated hardened anti-collapse policy


def _features(n_rows, split, encoder):
    """distilroberta embeddings of the candpool TRAIN prompts (cached alongside the matrix)."""
    cache = OUT / "enc_embed.npz"
    if cache.exists():
        H = np.load(cache, allow_pickle=True)["Htr"]
        if len(H) >= n_rows:
            return H[:n_rows]
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))[split][:n_rows]
    H = embed(encoder, prompts)
    np.savez(cache, Htr=H)
    return H


def _bo(rank, Mev, k):
    """Realized best-of-k: mean over eval prompts of max swing among the k top-ranked menu moves."""
    return float(np.mean([np.nan_to_num(Mev[i, rank[i, :k]], nan=-1e9).max() for i in range(len(Mev))]))


def _rank_random(K, n, rng):
    return np.array([rng.permutation(K)[:min(K, 8)] for _ in range(n)])


def _rank_oracle(Mev):
    return np.argsort(-np.nan_to_num(Mev, nan=-1e9), axis=1)


def _menu(family, order_tr, K, C, rng):
    if family == "greedy":
        return list(order_tr[:K])
    return list(rng.choice(C, K, replace=False))          # random subset of the pool


def sweep(M, H, Kgrid, families, seeds, n_pca, kshow=(1, 2, 4), include_null=True):
    """include_null: the null move (base, swing=0) is a real menu arm competing for a generation slot.
    Deployed arm set = selected moves (+ null). best-of-k picks k of them — so the router can spend a slot
    on base when the moves are risky. This matches the selection objective (which already floors at 0) and
    is the honest baseline vs BoN. NOTE: on the mean-swing matrix base=0 exactly; base-SAMPLE variance
    (what real BoN exploits) lives in the live banner, not here."""
    C = M.shape[1]
    Mn = np.concatenate([M, np.zeros((len(M), 1))], axis=1) if include_null else M   # null = last col (swing 0)
    # per (family, K): realized bandit@k, random@k, oracle@k across seeds
    res = {(fam, K): {f"bandit@{k}": [] for k in kshow} for fam in families for K in Kgrid}
    for fam in families:
        for K in Kgrid:
            for k in kshow:
                res[(fam, K)][f"random@{k}"] = []
            res[(fam, K)]["oracle"] = []   # per-prompt menu max (oracle best-of-k = menu-max for any k>=1)
    for seed in range(seeds):
        idx = np.arange(len(M)); np.random.default_rng(seed).shuffle(idx)
        a = int(0.75 * len(idx)); tr, ev = idx[:a], idx[a:]
        Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], n_pca)
        order_tr, _ = _greedy_submodular(M[tr], min(max(Kgrid), C))    # selection on TRAIN only (honest)
        rng = np.random.default_rng(seed + 5)
        for fam in families:
            for K in Kgrid:
                cols = _menu(fam, order_tr, K, C, rng)
                if include_null:
                    cols = cols + [C]                                  # append null arm to the deployed menu
                D = len(cols)
                Mtr, Mev = Mn[np.ix_(tr, cols)], Mn[np.ix_(ev, cols)]
                rk_b = train_policy(Ztr, _R_from_M(Mtr), HP, seed)(Zev)
                rk_o = _rank_oracle(Mev)
                rk_r = _rank_random(D, len(ev), rng)
                for k in kshow:
                    res[(fam, K)][f"bandit@{k}"].append(_bo(rk_b, Mev, k))
                    res[(fam, K)][f"random@{k}"].append(_bo(rk_r, Mev, k))
                res[(fam, K)]["oracle"].append(_bo(rk_o, Mev, 2))   # = menu max (>=0 with null)
    return {kk: {m: np.array(v) for m, v in d.items()} for kk, d in res.items()}


def _harder(M, H, Kfull, seeds, n_pca, epochs_grid):
    """Does training the bandit HARDER rescue the full menu? (the 'keep-everything + train-harder' arm)"""
    C = M.shape[1]; out = {e: [] for e in epochs_grid}
    for seed in range(seeds):
        idx = np.arange(len(M)); np.random.default_rng(seed).shuffle(idx)
        a = int(0.75 * len(idx)); tr, ev = idx[:a], idx[a:]
        Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], n_pca)
        cols = list(range(min(Kfull, C)))
        Mtr, Mev = M[np.ix_(tr, cols)], M[np.ix_(ev, cols)]
        for e in epochs_grid:
            hp = dict(HP, epochs=e)
            rk = train_policy(Ztr, _R_from_M(Mtr), hp, seed)(Zev)
            out[e].append(_bo(rk, Mev, 2))
    return {e: np.array(v) for e, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--split", default="train_large")
    args = ap.parse_args()
    t0 = time.time()
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    M = sw["M"]; C = M.shape[1]
    H = _features(len(M), args.split, args.encoder)
    Kgrid = [k for k in (2, 4, 8, 16, 32, 64, 128, C) if k <= C]
    if C not in Kgrid:
        Kgrid.append(C)
    Kgrid = sorted(set(Kgrid))
    R = sweep(M, H, Kgrid, ("greedy", "random"), args.seeds, args.n_pca, include_null=True)
    R0 = sweep(M, H, Kgrid, ("greedy",), args.seeds, args.n_pca, include_null=False)  # contrast: no base arm
    hard = _harder(M, H, C, args.seeds, args.n_pca, [60, 200, 400])

    def cell(fam, K, m):
        return f"{R[(fam, K)][m].mean():+.3f}"

    rows = ["# P1 menu-size sweep — does small-basis selection earn its keep? (offline, FREE)\n",
            f"Candpool swing matrix {M.shape[0]}×{C}, cached-swing reward oracle, {args.seeds} seeds, 75/25 "
            f"split, bandit-as-ranker (epochs={HP['epochs']}), PCA-{args.n_pca} {args.encoder} features. "
            "Realized best-of-2 vs random-routing floor vs oracle ceiling, as the menu grows. "
            "**The null move (base, swing 0) is an explicit menu arm competing for a generation slot.**\n",
            "## Greedy menu (submodular top-K, selected on the train split)",
            "_oracle = per-prompt menu max (with null: floored at base) = the best-of-k ceiling for ANY k. "
            "bandit@2(no-null) = the old arm without a base fallback, for contrast._",
            "| K | bandit@2 | bandit@2 (no-null) | random@2 | oracle ceiling |", "|---|---|---|---|---|"]
    for K in Kgrid:
        rows.append(f"| {K} | {cell('greedy',K,'bandit@2')} | {R0[('greedy',K)]['bandit@2'].mean():+.3f} | "
                    f"{cell('greedy',K,'random@2')} | {cell('greedy',K,'oracle')} |")
    rows += ["", "## Random menu (a K-subset of the pool, no selection)",
             "| K | bandit@2 | random@2 | oracle ceiling |", "|---|---|---|---|"]
    for K in Kgrid:
        rows.append(f"| {K} | {cell('random',K,'bandit@2')} | {cell('random',K,'random@2')} | "
                    f"{cell('random',K,'oracle')} |")
    rows += ["", "## Realized bandit best-of-k vs menu size (greedy menu)",
             "| K | bandit@1 | bandit@2 | bandit@4 |", "|---|---|---|---|"]
    for K in Kgrid:
        rows.append(f"| {K} | {cell('greedy',K,'bandit@1')} | {cell('greedy',K,'bandit@2')} | "
                    f"{cell('greedy',K,'bandit@4')} |")
    rows += ["", f"## Keep-everything + train harder (full menu K={C}, greedy=random=all cols)",
             "| epochs | bandit@2 |", "|---|---|"]
    for e in (60, 200, 400):
        rows.append(f"| {e} | {hard[e].mean():+.3f} |")

    # decision stats: is realized bandit@2 at the best small K different from the full menu?
    best_small = max((K for K in Kgrid if K <= 16), key=lambda K: R[("greedy", K)]["bandit@2"].mean())
    d = R[("greedy", best_small)]["bandit@2"] - R[("greedy", C)]["bandit@2"]
    lo, hi = _boot(d)
    argmax_K = max(Kgrid, key=lambda K: R[("greedy", K)]["bandit@2"].mean())
    rows += ["", "## Reading",
             f"- realized bandit@2 is maximized at **K={argmax_K}** (greedy menu).",
             f"- best small menu (K={best_small}) − full menu (K={C}): **{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]** ⇒ "
             + ("selecting a small basis beats keeping the pool at equal deploy compute — **stage (ii) earns its keep**."
                if lo > 0 else
                "no significant loss from keeping the pool — **selection is ~a no-op; keep-large + bandit is viable**."
                if hi > 0 else
                "the FULL menu beats the small one — **keep the pool; drop aggressive selection**."),
             f"- oracle ceiling vs bandit@2 at K={C}: {R[('greedy',C)]['oracle'].mean():+.3f} vs "
             f"{R[('greedy',C)]['bandit@2'].mean():+.3f} — the routing gap on the full menu (info-limit + "
             "exploration cost); if bandit@2 collapses toward random@2 as K grows, the big menu is unrouteable.",
             f"- train-harder at K={C}: {hard[60].mean():+.3f} (60ep) → {hard[400].mean():+.3f} (400ep) ⇒ "
             + ("more training DOES help the full menu."
                if hard[400].mean() - hard[60].mean() > 0.03 else
                "more training does NOT rescue the full menu (the ceiling is routing/info, not optimization)."),
             "", "_Mechanism note: a larger menu raises oracle@k (more headroom) but the bandit must explore "
             "more arms with the same data; the bandit@2−random@2 gap vs K shows whether routing keeps up._"]
    (REPO_ROOT / "basis").mkdir(exist_ok=True)
    rpt = REPO_ROOT / "basis" / "s1_menu_size_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\n[{time.time()-t0:.0f}s wall] report -> {rpt}")


if __name__ == "__main__":
    main()
