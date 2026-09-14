"""MENTOR-BA / M0 — cheapest gate BEFORE any move discovery or selection. OLMo (frozen) answers WildChat
coding prompts under the v2 mentoring guidebook; the Qwen analytic gold judge scores them. Two checks only:

  headroom : OLMo+guidebook is NOT saturated on the analytic rubric (mean overall well below 21, a real low
             tail, meaningful solution-dump rate) -> the rich guidebook leaves room for a move to help.
  validity : on constructed appropriate-vs-inappropriate control pairs, the gold's summed score is higher for
             the appropriate response on >=0.90 of pairs (and C1 discriminates) -> Qwen is a valid gold on
             the v2 rubric. Else escalate the judge before building discovery.

No discovery, no moves, no selection here. Phases keep OLMo and Qwen off the GPU together.

  python src/mentor_m0.py --phase gen   --base-config configs/base_olmo3sft.yaml
  python src/mentor_m0.py --phase judge
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_judge import load_judge
from mt_swing import _gen_multiturn
from analytic_judge import score_analytic
from mentor_c0 import _coding_prompts          # reuse the WildChat coding-prompt filter

CFG = "configs/mentor_moves_v2.yaml"


def _cfg():
    return yaml.safe_load(open(REPO_ROOT / CFG))


def _out():
    d = REPO_ROOT / "results" / "mentor_m0"; d.mkdir(parents=True, exist_ok=True); return d


def phase_gen(c, base_cfg):
    O = _out(); gen = c["gen"]
    P = _coding_prompts(c)[:c["m0"]["n_prompts"]]
    json.dump(P, open(O / "prompts.json", "w"))
    print(f"coding prompts: {len(P)}", flush=True)
    ctrls = c["controls"]
    device = resolve_device(base_cfg); t0 = time.time()
    model, tok = load_base(base_cfg, device); B = gen.get("batch", 4)

    # main: OLMo under the guidebook system prompt, m samples
    with open(O / "gen_main.jsonl", "w") as f:
        for rep in range(gen["m_samples"]):
            for i in range(0, len(P), B):
                chunk = P[i:i + B]
                convs = [[{"role": "user", "content": x["prompt"]}] for x in chunk]
                for x, a in zip(chunk, _gen_multiturn(model, tok, convs, gen, c["guidebook"])):
                    if a.strip():
                        f.write(json.dumps({"ci": x["ci"], "s": rep, "text": a}) + "\n")
            f.flush()
        print("  main gen done", flush=True)

    # controls: appropriate ('good') vs inappropriate ('bad') response per scenario; system=instruction,
    # NO guidebook (the instruction drives the polarity). Distinct system per scenario -> generate singly.
    with open(O / "gen_ctrl.jsonl", "w") as f:
        for j, s in enumerate(ctrls):
            for pol in ("good", "bad"):
                a = _gen_multiturn(model, tok, [[{"role": "user", "content": s["prompt"]}]], gen, s[pol])[0]
                if a.strip():
                    f.write(json.dumps({"cid": j, "pol": pol, "text": a}) + "\n")
        print("  control gen done", flush=True)
    print(log_cost("MENTOR", "m0_gen", time.time() - t0, device,
                   notes=f"{len(P)} prompts x {gen['m_samples']} + {len(ctrls)} control pairs"))


def phase_judge(c):
    O = _out(); rubric = c["rubric"]; mnt = c["judge"]["max_new_tokens"]
    P = {x["ci"]: x["prompt"] for x in json.load(open(O / "prompts.json"))}
    main = defaultdict(list)
    for l in open(O / "gen_main.jsonl"):
        r = json.loads(l); main[r["ci"]].append(r["text"])
    ctrl = defaultdict(dict)
    for l in open(O / "gen_ctrl.jsonl"):
        r = json.loads(l); ctrl[r["cid"]][r["pol"]] = r["text"]

    device = resolve_device(load_config("configs/base_olmo3sft.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device)
    jb = c["judge"].get("batch", 4)

    # headroom: analytic score of the guidebook-conditioned responses
    items = [(P[ci], t) for ci in P for t in main.get(ci, [])]
    sc = score_analytic(mdl, tok, items, rubric, mnt, batch=jb)
    parse = float(np.mean([s is not None for s in sc]))
    ov = np.array([s["overall"] for s in sc if s], float)
    cats = {k: np.array([s[k] for s in sc if s], float) for k in ["C1", "C2", "C3", "C4", "C5"]}

    # validity: appropriate ('good') vs inappropriate ('bad') per control scenario
    citems, cidx = [], []
    for j in sorted(ctrl):
        for pol in ("good", "bad"):
            if pol in ctrl[j]:
                citems.append((c["controls"][j]["prompt"], ctrl[j][pol])); cidx.append((j, pol))
    csc = score_analytic(mdl, tok, citems, rubric, mnt, batch=jb)
    byj = defaultdict(dict)
    for (j, pol), s in zip(cidx, csc):
        if s:
            byj[j][pol] = s
    pairs = [(v["good"], v["bad"]) for v in byj.values() if "good" in v and "bad" in v]
    validity = float(np.mean([g["overall"] > b["overall"] for g, b in pairs])) if pairs else float("nan")
    c1_good = float(np.mean([g["C1"] for g, b in pairs])) if pairs else float("nan")
    c1_bad = float(np.mean([b["C1"] for g, b in pairs])) if pairs else float("nan")

    np.savez(O / "mentor_m0.npz", overall=ov, parse=parse, validity=validity,
             ctrl_good=np.array([g["overall"] for g, b in pairs], float),
             ctrl_bad=np.array([b["overall"] for g, b in pairs], float),
             **{k: cats[k] for k in cats})
    print(log_cost("MENTOR", "m0_judge", time.time() - t0, device, notes=f"{len(P)} prompts, {len(pairs)} controls"))

    dump_rate = float(np.mean(cats["C2"] <= 0))       # C2==0 == dumped a full solution / no teaching
    print("\n=== MENTOR-BA / M0 (gold = Qwen analytic v2 rubric) ===")
    print(f"judge parse-rate: {parse:.0%}   (n scored {len(ov)})")
    print(f"headroom : OLMo+guidebook overall {ov.mean():.1f}/21  "
          f"(frac>=18 {np.mean(ov>=18):.0%}, frac<=10 {np.mean(ov<=10):.0%}); dump-rate (C2=0) {dump_rate:.0%}")
    print("  per-category means:  " + "  ".join(f"{k} {cats[k].mean():.2f}" for k in ["C1", "C2", "C3", "C4"])
          + f"  C5 {cats['C5'].mean():.2f}")
    print(f"validity : gold prefers APPROPRIATE over INAPPROPRIATE = {validity:.2f}  (need >=0.90); "
          f"C1 good {c1_good:.2f} vs bad {c1_bad:.2f}")
    g_head = (ov.mean() < 17) and (np.mean(ov >= 18) < 0.6)
    g_val = validity >= 0.90
    print(f"\nGREEN  headroom={g_head}  validity={g_val}  ->  "
          f"{'PROCEED to discovery (M1)' if (g_head and g_val) else 'STOP: report the failing check'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--base-config", default="configs/base_olmo3sft.yaml")
    args = ap.parse_args()
    c = _cfg()
    if args.phase == "gen":
        phase_gen(c, load_config(args.base_config))
    else:
        phase_judge(c)


if __name__ == "__main__":
    main()
