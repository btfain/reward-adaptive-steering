"""M1b analysis — does observing the user's reaction (u1,a1,u2) make the right MOVE predictable, breaking
the single-turn ~18% capture ceiling? On the multi-turn swing matrix (mt_swing.py), reusing the same
bandit-as-ranker + capture machinery:

  1. capture      — realized bandit best-of-2 vs random floor vs oracle ceiling; captured = (bandit-random)/
                    (oracle-random). Compare to the single-turn ~18%.
  2. heterogeneity— does the argmax move VARY by context, or does one dominate (the RM-era failure)?
  3. separation   — how much do the moves lift over NULL (no intervention)?
  4. length guard — corr(judge score, response length): is our custom rubric length-decoupled (unlike the RM)?

    python src/mt_capture.py --config configs/mt_swing_wildchat_v1.yaml --seeds 16
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from models import REPO_ROOT
from menu_size_sweep import _bo, _rank_oracle, _rank_random
from bandit_ranker_val import train_policy
from router_bandit import _pca, _R_from_M, _boot
from bakeoff_rankers import embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n_pca", type=int, default=40)
    args = ap.parse_args()
    c = yaml.safe_load(open(REPO_ROOT / args.config))
    O = REPO_ROOT / "results" / c["tag"]
    d = np.load(O / "swing.npz", allow_pickle=True)
    M = d["M"]; names = list(d["move_names"]); ctx = list(d["contexts"])
    full = ~np.isnan(M).any(1)                                  # keep contexts with all moves scored
    M, ctx = M[full], [ctx[i] for i in np.where(full)[0]]
    n, K = M.shape
    null_col = names.index("null") if "null" in names else 0

    # features: e5 on the concatenated (u1,a1,u2)
    H = embed(c["router"]["encoder"], ctx, max_len=512)

    # 1) capture (bandit best-of-2), multi-seed CV
    cap, real, rnd, orc = [], [], [], []
    for seed in range(args.seeds):
        idx = np.arange(n); np.random.default_rng(seed).shuffle(idx)
        a = int(0.75 * n); tr, ev = idx[:a], idx[a:]
        Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], min(args.n_pca, H.shape[1]))
        Mev = M[ev]
        rk = train_policy(Ztr, _R_from_M(M[tr]), dict(hidden=0, lr=0.02, beta=0.1, batch=64, epochs=60), seed)(Zev)
        rng = np.random.default_rng(seed + 5)
        rb = _bo(rk, Mev, 2); rr = _bo(_rank_random(K, len(ev), rng), Mev, 2); ro = _bo(_rank_oracle(Mev), Mev, 2)
        real.append(rb); rnd.append(rr); orc.append(ro)
        cap.append((rb - rr) / (ro - rr) if ro > rr else 0.0)
    cap = np.array(cap); clo, chi = _boot(cap)

    # 2) heterogeneity: argmax move per context (mean-score), 3) separation vs null
    argmax = [names[j] for j in np.argmax(M, 1)]
    win = Counter(argmax); top_share = win.most_common(1)[0][1] / n
    best = M.max(1); nullv = M[:, null_col]
    second = np.sort(M, 1)[:, -2]
    beats_null = float((best > nullv + 1e-9).mean())
    permove = [(names[k], float(np.nanmean(M[:, k] - nullv))) for k in range(K)]
    permove.sort(key=lambda x: -x[1])

    # 4) length decoupling
    sf, lf = d["scores_flat"], d["lens_flat"]
    lcorr = float(np.corrcoef(sf, lf)[0, 1]) if len(sf) > 2 else float("nan")

    rows = [f"# M1b — multi-turn capture: does observing u2 break the single-turn ceiling? — {c['tag']}\n",
            f"{n} contexts (all {K} moves scored), Prometheus+rubric reward, e5 router features, {args.seeds} "
            "seeds. Single-turn reference: capture ~18%, argmax dominated by one generic move.\n",
            "## 1) Capture (bandit best-of-2)",
            f"- realized {np.mean(real):+.3f} · random {np.mean(rnd):+.3f} · oracle {np.mean(orc):+.3f} (scale 1-5)",
            f"- **headroom captured = {100*cap.mean():.0f}% [{100*clo:.0f}%, {100*chi:.0f}%]** "
            + ("=> BEATS the single-turn ~18% ceiling: observing u2 makes the move predictable."
               if clo > 0.18 else "=> not clearly above the single-turn ~18% — observing u2 did not unlock routing here."),
            "", "## 2) Heterogeneity (argmax move per context)",
            f"- top move '{win.most_common(1)[0][0]}' wins {100*top_share:.0f}% of contexts; "
            f"{len([k for k in win if win[k]>0])}/{K} moves win at least one context.",
            "  " + ", ".join(f"{k}:{win.get(k,0)}" for k in names),
            ("- => moves are CONTEXT-DEPENDENT (no single dominant move) — the routable structure the RM lacked."
             if top_share < 0.5 else "- => one move still dominates — heterogeneity did not materialize."),
            "", "## 3) Separation over NULL (no intervention)",
            f"- some move beats null in {100*beats_null:.0f}% of contexts; mean(best-null) {np.mean(best-nullv):+.2f}, "
            f"mean(best-2nd) {np.mean(best-second):+.2f} (score points).",
            "  per-move mean lift vs null: " + ", ".join(f"{nm}{v:+.2f}" for nm, v in permove),
            "", "## 4) Length decoupling (the folded-in guard)",
            f"- corr(judge score, response length) = **{lcorr:+.2f}** "
            + ("(low => our rubric is length-decoupled, unlike the style-biased RM)."
               if abs(lcorr) < 0.2 else "(HIGH => residual verbosity bias; distrust this run, fix the rubric)."),
            ]
    rpt = REPO_ROOT / "basis" / f"rb_{c['tag']}_capture_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
