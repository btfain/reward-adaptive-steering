"""P1(i) evaluation — is the auto-generated candidate pool as good as (or better than) the hand-curated
one? Runs greedy submodular selection on the combined-pool swing matrix restricted to AUTO-only,
CURATED-only, and COMBINED columns, and compares the value-vs-K curves (holding prompts + selection
algorithm fixed). Also reports how the COMBINED basis is composed (auto vs curated) — does the auto pool
contribute moves the curated pool lacks? Offline; run after the candpool swing matrix is assembled.

    python src/candidate_eval.py --tag candpool_7b
"""

import argparse
import json
from pathlib import Path

import numpy as np

from prompt_basis import _greedy_submodular

REPO_ROOT = Path(__file__).resolve().parent.parent


def _value(M, order):                                       # f(S)=Σ_x max(0, max_{p∈S} swing)
    out = []
    for k in range(1, len(order) + 1):
        cols = M[:, order[:k]]
        best = np.nan_to_num(cols, nan=-1e9).max(1)
        out.append(float(np.maximum(0.0, best).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="candpool_7b")
    ap.add_argument("--combined", default="configs/candidates_medoid_combined_7b.txt")
    ap.add_argument("--K", type=int, default=16)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    M = sw["M"]                                             # (n_prompts, C)
    n_auto = None
    for l in open(REPO_ROOT / args.combined):
        if l.startswith("#") and "n_auto=" in l:
            n_auto = int(l.split("n_auto=")[1].split()[0]); break
    if n_auto is None:
        raise RuntimeError("n_auto marker not found in combined candidate file header")
    C = M.shape[1]
    auto_cols = list(range(n_auto)); cur_cols = list(range(n_auto, C))

    def sel(cols):
        sub = M[:, cols]
        order_local, _curve = _greedy_submodular(sub, min(args.K, len(cols)))   # returns (order, curve)
        return [cols[i] for i in order_local]              # map back to global column indices

    o_auto, o_cur, o_comb = sel(auto_cols), sel(cur_cols), sel(list(range(C)))
    v_auto, v_cur, v_comb = _value(M, o_auto), _value(M, o_cur), _value(M, o_comb)
    Kshow = min(args.K, len(o_comb))
    comp = ["auto" if c < n_auto else "curated" for c in o_comb]

    rows = [f"# P1(i) — candidate pool evaluation (auto vs curated vs combined) — {args.tag}\n",
            f"Greedy submodular value f(S)=Σ_x max(0,max swing) on the SAME {M.shape[0]} prompts; only the "
            f"candidate pool varies. Pool: {n_auto} auto + {C-n_auto} curated = {C}.\n",
            "| K | auto-only | curated-only | combined |", "|---|---|---|---|"]
    for k in range(Kshow):
        rows.append(f"| {k+1} | {v_auto[k]:+.3f} | {v_cur[k]:+.3f} | {v_comb[k]:+.3f} |")
    na = comp[:Kshow].count("auto")
    rows += ["", f"## Combined basis composition (top-{Kshow}): {na} auto / {Kshow-na} curated",
             "  " + "  ".join(f"{k+1}:{comp[k]}" for k in range(Kshow)),
             "", "## Reading",
             f"- auto-only vs curated-only at K=8: {v_auto[min(7,Kshow-1)]:+.3f} vs {v_cur[min(7,Kshow-1)]:+.3f} ⇒ "
             + ("auto pool ≥ curated (automation matches/beats hand-crafting)."
                if v_auto[min(7,Kshow-1)] >= v_cur[min(7,Kshow-1)] - 0.02 else
                "auto pool below curated — improve the generator (more calls / stronger model / better signal)."),
             f"- combined − max(auto,curated) at K=8: "
             f"{v_comb[min(7,Kshow-1)] - max(v_auto[min(7,Kshow-1)], v_cur[min(7,Kshow-1)]):+.3f} ⇒ "
             "does mixing pools add coverage (complementarity)?",
             f"- combined basis draws {na}/{Kshow} from auto ⇒ the auto pool contributes "
             + ("substantially." if na >= Kshow // 2 else "little; curated dominates.")]
    rpt = REPO_ROOT / "basis" / f"s1_candidate_eval_{args.tag}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
