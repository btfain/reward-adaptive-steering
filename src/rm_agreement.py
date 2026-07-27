"""A2 RM-variance probe: re-score cached completions with the second RM.

The A2 pilot cached every completion (results/headroom/headroom_log.jsonl). This
scores those SAME completions with the alternate RM (Skywork-V2-Llama, vs the
primary Skywork-V2-Qwen3) and asks: do two independent RMs agree on the SIGN of
each per-condition effect? Disagreement / joint-flatness is the signature of the
"RM is the problem, or the variants are a genuine toss-up" branch; sign-agreement
on the effects that move is evidence the headroom is real preference.

No generation — only scoring the cached text, so it is fast and cheap.

Outputs: results/headroom/headroom_rm2.jsonl (rm2 per cached row),
basis/rm_agreement.md.
"""

import json
import time

import numpy as np
from models import REPO_ROOT, load_config, load_rm, log_cost, resolve_device, rm_score

OUT = REPO_ROOT / "results" / "headroom"
BASIS = REPO_ROOT / "basis"


def score_alt():
    cfg = load_config()
    device = resolve_device(cfg)
    alt = {**cfg, "reward_model": cfg["reward_model_alt"]}
    rm, rm_tok = load_rm(alt, device)
    rows = [json.loads(l) for l in open(OUT / "headroom_log.jsonl")]
    out = OUT / "headroom_rm2.jsonl"
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"scoring {len(rows) - done} completions with {cfg['reward_model_alt']}")
    with open(out, "a") as f:
        for r in rows[done:]:
            f.write(json.dumps({
                "model": r["model"], "condition": r["condition"], "seed": r["seed"],
                "prompt": r["prompt"], "rm2": rm_score(rm, rm_tok, r["prompt"], r["completion"]),
            }) + "\n")
            f.flush()
    return cfg


def report(cfg):
    r1 = [json.loads(l) for l in open(OUT / "headroom_log.jsonl")]
    r2 = {(x["model"], x["condition"], x["seed"], x["prompt"]): x["rm2"]
          for x in (json.loads(l) for l in open(OUT / "headroom_rm2.jsonl"))}
    lines = ["# A2 RM-variance probe — Skywork-V2-Qwen3-0.6B vs -Llama-3.2-1B\n",
             f"Both RMs score the SAME {len(r1)} cached pilot completions. RM1 = "
             f"{cfg['reward_model'].split('/')[-1]}, RM2 = {cfg['reward_model_alt'].split('/')[-1]}.\n"]
    for mk in sorted({x["model"] for x in r1}):
        mr = [x for x in r1 if x["model"] == mk]
        n1 = {(x["prompt"], x["seed"]): x["rm"] for x in mr if x["condition"] == "none"}
        n2 = {(x["prompt"], x["seed"]): r2[(mk, "none", x["seed"], x["prompt"])]
              for x in mr if x["condition"] == "none"}
        conds = [c for c in dict.fromkeys(x["condition"] for x in mr) if c != "none"]

        # per-(prompt,seed) paired deltas under each RM -> correlation
        d1all, d2all, rows = [], [], []
        for c in conds:
            d1, d2 = [], []
            for x in mr:
                if x["condition"] != c:
                    continue
                k = (x["prompt"], x["seed"])
                if k in n1:
                    d1.append(x["rm"] - n1[k])
                    d2.append(r2[(mk, c, x["seed"], x["prompt"])] - n2[k])
            d1all += d1; d2all += d2
            rows.append((c, float(np.mean(d1)), float(np.mean(d2))))
        r = float(np.corrcoef(d1all, d2all)[0, 1])
        # sign agreement on per-condition MEAN effects
        agree = np.mean([np.sign(a) == np.sign(b) for _, a, b in rows])
        # agreement weighted to the conditions that actually move RM1 (|mean|>0.3)
        movers = [(a, b) for _, a, b in rows if abs(a) > 0.3]
        agree_mv = np.mean([np.sign(a) == np.sign(b) for a, b in movers]) if movers else float("nan")

        lines.append(f"\n## {mk}\n")
        lines.append(f"- per-(prompt,seed) ΔRM correlation RM1↔RM2: **{r:+.2f}**")
        lines.append(f"- per-condition mean-effect sign agreement: **{agree:.0%}** "
                     f"(all {len(rows)}), **{agree_mv:.0%}** on {len(movers)} movers (|ΔRM1|>0.3)")
        lines.append("\n| condition | ΔRM1 | ΔRM2 | agree |")
        lines.append("|---|---|---|---|")
        for c, a, b in sorted(rows, key=lambda t: -t[1]):
            lines.append(f"| {c} | {a:+.2f} | {b:+.2f} | {'✓' if np.sign(a)==np.sign(b) else '✗'} |")
    BASIS.mkdir(exist_ok=True)
    (BASIS / "rm_agreement.md").write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if not l.startswith("|")))
    print(f"\nreport -> {BASIS / 'rm_agreement.md'}")


def main():
    t0 = time.time()
    cfg = score_alt()
    report(cfg)
    print(log_cost("A2", "rm_agreement", time.time() - t0, resolve_device(cfg)))


if __name__ == "__main__":
    main()
