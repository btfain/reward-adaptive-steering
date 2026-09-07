"""Synthetic notation positive-control — the router-capability gate. Reward is NOISE-FREE and the oracle
move is KNOWN (each prompt's type). The baseline-to-beat is `full_rubric` (all rules in one system prompt).

Reports realized top-1 (route ONE bundle per prompt):
  null · full_rubric(baseline) · blind(best fixed move) · routed(bandit & ridge) · oracle(type bundle)
plus routing ACCURACY (does the router pick the correct type bundle?).

If routed >= full_rubric: externalized context-routing beats monolithic instruction-stuffing.
If routed can't even beat blind/reach oracle: the router/features/algorithm is the bottleneck (fix that).

    python src/synth_capture.py --config configs/synth_notation_v1.yaml --seeds 16
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from models import REPO_ROOT
from bakeoff_rankers import embed
from bandit_ranker_val import ridge_rank, train_policy
from router_bandit import _pca, _R_from_M, _boot


def _top1(rank, Mev):
    return float(np.mean([Mev[i, rank[i, 0]] for i in range(len(Mev))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n_pca", type=int, default=40)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    O = REPO_ROOT / "results" / cfg["tag"]
    d = np.load(O / "synth.npz", allow_pickle=True)
    M = d["M"]; names = list(d["move_names"]); ptype = list(d["prompt_types"]); ctx = list(d["contexts"])
    ok = ~np.isnan(M).any(1)
    M = M[ok]; ptype = [ptype[i] for i in np.where(ok)[0]]; ctx = [ctx[i] for i in np.where(ok)[0]]
    n = len(M)
    null_c = names.index("null"); fr_c = names.index("full_rubric")
    bundle_col = {t: names.index(t) for t in ptype}                 # type name == its bundle move name
    oracle_col = np.array([names.index(t) for t in ptype])
    H = embed(cfg["router"]["encoder"], ctx, max_len=512)

    R = {k: [] for k in ("null", "full_rubric", "blind", "bandit", "ridge", "oracle", "acc_bandit", "acc_ridge")}
    for seed in range(args.seeds):
        idx = np.arange(n); np.random.default_rng(seed).shuffle(idx)
        a = int(0.75 * n); tr, ev = idx[:a], idx[a:]
        Ztr, Zev = _pca(H[tr], [H[tr], H[ev]], min(args.n_pca, H.shape[1]))
        Mev = M[ev]
        rk_b = train_policy(Ztr, _R_from_M(M[tr]), dict(hidden=0, lr=0.02, beta=0.1, batch=64, epochs=60), seed)(Zev)
        rk_r = ridge_rank(H[tr], M[tr], H[ev])
        blind_mv = int(np.argmax(M[tr].mean(0)))                    # best FIXED move from train
        R["null"].append(Mev[:, null_c].mean()); R["full_rubric"].append(Mev[:, fr_c].mean())
        R["blind"].append(Mev[:, blind_mv].mean())
        R["bandit"].append(_top1(rk_b, Mev)); R["ridge"].append(_top1(rk_r, Mev))
        R["oracle"].append(float(np.mean(Mev[np.arange(len(ev)), oracle_col[ev]])))
        R["acc_bandit"].append(float(np.mean(rk_b[:, 0] == oracle_col[ev])))
        R["acc_ridge"].append(float(np.mean(rk_r[:, 0] == oracle_col[ev])))
    m = {k: np.array(v) for k, v in R.items()}
    gain = m["bandit"] - m["full_rubric"]; glo, ghi = _boot(gain)
    blind_name = names[int(np.argmax(M.mean(0)))]

    rows = [f"# Synthetic notation positive-control — router capability (noise-free, known oracle) — {cfg['tag']}\n",
            f"{n} typed prompts, {len(names)} moves, deterministic reward, e5 features, {args.seeds} seeds, "
            "realized top-1 (route ONE bundle/prompt). Baseline-to-beat = full_rubric (all rules, one prompt).\n",
            "| arm | realized satisfaction |", "|---|---|",
            f"| null (no rules) | {m['null'].mean():.3f} |",
            f"| **full_rubric (baseline)** | **{m['full_rubric'].mean():.3f}** |",
            f"| blind (best fixed move = {blind_name}) | {m['blind'].mean():.3f} |",
            f"| routed — ridge | {m['ridge'].mean():.3f} |",
            f"| **routed — bandit** | **{m['bandit'].mean():.3f}** |",
            f"| oracle (type bundle) | {m['oracle'].mean():.3f} |",
            "",
            f"- routing accuracy (picks correct type bundle): bandit {100*m['acc_bandit'].mean():.0f}%, "
            f"ridge {100*m['acc_ridge'].mean():.0f}% (chance {100/ (len(names)):.0f}%).",
            f"- **routed(bandit) − full_rubric = {gain.mean():+.3f} [{glo:+.3f}, {ghi:+.3f}]** "
            + ("=> routed does not beat the full-rubric prompt here." if ghi <= 0 else
               ("=> routing MATERIALLY beats monolithic instruction-stuffing." if gain.mean() >= 0.10 else
                "=> routing beats the full-rubric prompt but only MARGINALLY: a strong model self-routes ~24 "
                "conditional rules well, so overload barely bites. Scale up constraints / use a weaker "
                "instruction-follower / add conflicting rules to widen the gap.")),
            f"- router vs oracle gap = {m['oracle'].mean()-m['bandit'].mean():+.3f} "
            "(if large despite noise-free known structure => the ROUTER/features/algorithm is the bottleneck)."]
    rpt = REPO_ROOT / "basis" / f"rb_{cfg['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
