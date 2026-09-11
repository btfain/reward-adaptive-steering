"""S1-NPOV / N0 — the cheap precondition gate BEFORE any move discovery. Qwen-7B is the held-out GOLD judge;
OLMo (frozen) is the base, revising biased sentences under the NPOV guidebook system prompt. Three checks:

  N0.1 headroom : OLMo's guidebook-conditioned revisions still fall short of the HUMAN neutralizations
                  (a real gap, and not already saturated) -> the domain leaves room for moves.
  N0.2 validity : the gold judge prefers the HUMAN-neutral over the BIASED original (target >= 0.75) ->
                  Qwen-7B is a usable NPOV gold judge AT ALL (doubles as the judge-vs-human check).
  N0.3 variance : real spread in adherence across samples -> exploitable heterogeneity for discovery.

GREEN (all three) unlocks N1 (reward-driven move discovery). Phases keep OLMo and Qwen off the GPU together.

  python src/npov_n0.py --phase data
  python src/npov_n0.py --phase gen   --base-config configs/base_olmo3sft.yaml
  python src/npov_n0.py --phase judge
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_judge import load_judge, prefer_batch, score_batch
from mt_swing import _gen_multiturn
import npov

CFG = "configs/npov_n0.yaml"


def _cfg():
    return yaml.safe_load(open(REPO_ROOT / CFG))


def _out():
    d = REPO_ROOT / "results" / "npov_n0"; d.mkdir(parents=True, exist_ok=True); return d


def phase_data(c):
    O = _out(); s = npov.split(c)
    json.dump(s, open(O / "split.json", "w"))
    print(f"WNC eval={len(s['eval'])} discover={len(s['discover'])}  (source: {c['data']['wnc_subpath']})")
    for r in s["eval"][:3]:
        print("  BIASED :", r["biased"][:160])
        print("  NEUTRAL:", r["neutral"][:160], "\n")


def phase_gen(c, base_cfg):
    O = _out(); ev = json.load(open(O / "split.json"))["eval"]; gen = c["gen"]
    device = resolve_device(base_cfg); t0 = time.time()
    model, tok = load_base(base_cfg, device)
    B = 16
    with open(O / "gen.jsonl", "w") as f:
        for rep in range(gen["m_samples"]):
            for i in range(0, len(ev), B):
                chunk = ev[i:i + B]
                convs = [[{"role": "user", "content": npov.neutralize_instruction(c, r["biased"])}]
                         for r in chunk]
                for r, a in zip(chunk, _gen_multiturn(model, tok, convs, gen, c["guidebook"])):
                    if a.strip():
                        f.write(json.dumps({"id": r["id"], "s": rep, "text": a}) + "\n")
            f.flush(); print(f"  gen sample {rep} done", flush=True)
    print(log_cost("NPOV", "n0_gen", time.time() - t0, device, notes=f"{len(ev)} eval x {gen['m_samples']}"))


def phase_judge(c):
    O = _out(); ev = {r["id"]: r for r in json.load(open(O / "split.json"))["eval"]}
    rubric = c["rubric"]; mnt = c["judge"]["max_new_tokens"]
    revs = defaultdict(list)
    for l in open(O / "gen.jsonl"):
        r = json.loads(l); revs[r["id"]].append(r["text"])

    device = resolve_device(load_config("configs/base_olmo3sft.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device)

    def ctx(rid):
        return [{"role": "user", "content": npov.neutralize_instruction(c, ev[rid]["biased"])}]

    ids = list(ev)

    # N0.1 headroom: absolute NPOV score of OLMo revisions vs the human-neutral ceiling (aligned, keep None)
    rev_keys = [rid for rid in ids for _ in revs.get(rid, [])]
    rev_items = [(ctx(rid), t, None) for rid in ids for t in revs.get(rid, [])]
    rev_raw = score_batch(mdl, tok, rev_items, rubric, mnt)
    by = defaultdict(list)
    for k, sc in zip(rev_keys, rev_raw):
        if sc is not None:
            by[k].append(sc)
    rev_sc = np.array([sc for v in by.values() for sc in v], float)
    hum_raw = score_batch(mdl, tok, [(ctx(rid), ev[rid]["neutral"], None) for rid in ids], rubric, mnt)
    hum_sc = np.array([x for x in hum_raw if x is not None], float)

    # N0.2 validity: does the gold judge prefer HUMAN-neutral over BIASED original? (both orders cancel bias)
    tasks, items = [], []
    for rid in ids:
        A, Bx = ev[rid]["neutral"], ev[rid]["biased"]
        tasks.append((rid, "A")); items.append((ctx(rid), A, Bx, None))
        tasks.append((rid, "B")); items.append((ctx(rid), Bx, A, None))
    res = prefer_batch(mdl, tok, items, rubric, mnt)
    vw = [int((tag == "A" and ab == "A") or (tag == "B" and ab == "B"))
          for (rid, tag), ab in zip(tasks, res) if ab is not None]
    validity = float(np.mean(vw)) if vw else float("nan")

    # N0.3 variance: per-prompt spread of adherence across the OLMo samples
    spreads = [max(v) - min(v) for v in by.values() if len(v) >= 2]
    frac_spread = float(np.mean([sp >= 2 for sp in spreads])) if spreads else float("nan")

    parse_rate = float(np.mean([x is not None for x in rev_raw]))
    np.savez(O / "npov_n0.npz", rev_sc=rev_sc, hum_sc=hum_sc, validity=validity,
             frac_spread=frac_spread, spreads=np.array(spreads, float), parse_rate=parse_rate)
    print(log_cost("NPOV", "n0_judge", time.time() - t0, device, notes=f"{len(ids)} eval prompts"))

    gap = hum_sc.mean() - rev_sc.mean()
    print("\n=== S1-NPOV / N0 (gold judge = Qwen-7B) ===")
    print(f"judge parse-rate: {parse_rate:.0%}   (low => the Prometheus-style [RESULT] format is failing on Qwen)")
    print(f"N0.1 headroom : OLMo rev {rev_sc.mean():.2f}/5 (frac>=4 {np.mean(rev_sc>=4):.0%}) vs "
          f"human {hum_sc.mean():.2f}/5  -> gap {gap:+.2f}")
    print(f"N0.2 validity : gold prefers HUMAN over BIASED = {validity:.2f}   (need >= 0.75)")
    print(f"N0.3 variance : frac prompts with sample spread >=2 = {frac_spread:.2f}   (need >= 0.30)")
    g1 = (gap >= 0.5) and (np.mean(rev_sc >= 4) < 0.85)
    g2 = validity >= 0.75
    g3 = frac_spread >= 0.30
    print(f"\nGREEN  N0.1={g1}  N0.2={g2}  N0.3={g3}  ->  "
          f"{'ALL GREEN: unlock N1 discovery' if (g1 and g2 and g3) else 'STOP: report the failing check'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["data", "gen", "judge"])
    ap.add_argument("--base-config", default="configs/base_olmo3sft.yaml")
    args = ap.parse_args()
    c = _cfg()
    if args.phase == "data":
        phase_data(c)
    elif args.phase == "gen":
        base_cfg = load_config(args.base_config)
        phase_gen(c, base_cfg)
    else:
        phase_judge(c)


if __name__ == "__main__":
    main()
