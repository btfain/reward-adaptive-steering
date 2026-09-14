"""S1-NPOV / N0 (paragraph-level) — the cheap precondition gate after redirecting off WNC's lexical sentence
edits. OLMo (frozen) writes NEUTRAL encyclopedic paragraphs on real contested topics under the NPOV
guidebook; Qwen-7B is the held-out GOLD judge. Three checks:

  N0.1 headroom : OLMo paragraphs are NOT saturated (frac>=4 < 0.85) and sit below the neutral-control
                  ceiling -> the domain leaves room for a move to improve.
  N0.2 validity : the gold judge prefers a NEUTRAL control paragraph over a blatantly SLANTED one, on the
                  same topic (>= 0.90, MASTER gate) -> Qwen is a usable NPOV judge on holistic paragraphs.
                  A floor control; if it fails, escalate the judge (API) before any discovery.
  N0.3 variance : per-topic adherence spread across samples -> exploitable heterogeneity for discovery.

GREEN (all three) unlocks N1 (move discovery). Phases keep OLMo and Qwen off the GPU together.

  python src/npov_par_n0.py --phase gen   --base-config configs/base_olmo3sft.yaml
  python src/npov_par_n0.py --phase judge
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

CFG = "configs/npov_paragraph_v1.yaml"


def _cfg():
    return yaml.safe_load(open(REPO_ROOT / CFG))


def _out():
    d = REPO_ROOT / "results" / "npov_par"; d.mkdir(parents=True, exist_ok=True); return d


def phase_gen(c, base_cfg):
    O = _out(); gen = c["gen"]
    topics = c["topics"][:c["data"]["n_prompts"]]
    ctrl = c["control_topics"]
    json.dump({"topics": topics, "control_topics": ctrl}, open(O / "prompts.json", "w"))
    device = resolve_device(base_cfg); t0 = time.time()
    model, tok = load_base(base_cfg, device); B = 16

    def run(instr, tops, m, system, arm):
        with open(O / f"gen_{arm}.jsonl", "w") as f:
            for rep in range(m):
                for i in range(0, len(tops), B):
                    chunk = tops[i:i + B]
                    convs = [[{"role": "user", "content": instr.format(topic=t)}] for t in chunk]
                    for t, a in zip(chunk, _gen_multiturn(model, tok, convs, gen, system)):
                        if a.strip():
                            f.write(json.dumps({"topic": t, "s": rep, "text": a}) + "\n")
                f.flush()
            print(f"  gen arm {arm} done ({len(tops)} topics x {m})", flush=True)

    run(c["task_instruction"], topics, gen["m_samples"], c["guidebook"], "main")
    run(c["control_neutral_instruction"], ctrl, 1, None, "ctrl_neutral")
    run(c["control_slanted_instruction"], ctrl, 1, None, "ctrl_slanted")
    print(log_cost("NPOV", "par_n0_gen", time.time() - t0, device,
                   notes=f"{len(topics)} topics x {gen['m_samples']} + {len(ctrl)} control pairs"))


def _read(path):
    d = defaultdict(list)
    for l in open(path):
        r = json.loads(l); d[r["topic"]].append(r["text"])
    return d


def phase_judge(c):
    O = _out(); P = json.load(open(O / "prompts.json"))
    topics, ctrl = P["topics"], P["control_topics"]
    rubric = c["rubric"]; mnt = c["judge"]["max_new_tokens"]
    main = _read(O / "gen_main.jsonl")
    cn = _read(O / "gen_ctrl_neutral.jsonl")
    cs = _read(O / "gen_ctrl_slanted.jsonl")

    device = resolve_device(load_config("configs/base_olmo3sft.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device)

    def ctx(topic):
        return [{"role": "user", "content": c["task_instruction"].format(topic=topic)}]

    # N0.1 headroom: adherence of the main paragraphs (aligned keys) + the neutral-control ceiling
    keys = [t for t in topics for _ in main.get(t, [])]
    items = [(ctx(t), x, None) for t in topics for x in main.get(t, [])]
    raw = score_batch(mdl, tok, items, rubric, mnt)
    by = defaultdict(list)
    for k, sc in zip(keys, raw):
        if sc is not None:
            by[k].append(sc)
    main_sc = np.array([sc for v in by.values() for sc in v], float)
    ceil_raw = score_batch(mdl, tok, [(ctx(t), cn[t][0], None) for t in ctrl if t in cn], rubric, mnt)
    ceil = np.array([x for x in ceil_raw if x is not None], float)

    # N0.2 validity: gold prefers NEUTRAL control over SLANTED control (both orders cancel position bias)
    tasks, pf = [], []
    for t in ctrl:
        if t in cn and t in cs:
            A, Bx = cn[t][0], cs[t][0]
            tasks.append((t, "A")); pf.append((ctx(t), A, Bx, None))
            tasks.append((t, "B")); pf.append((ctx(t), Bx, A, None))
    res = prefer_batch(mdl, tok, pf, rubric, mnt)
    vw = [int((tag == "A" and ab == "A") or (tag == "B" and ab == "B"))
          for (t, tag), ab in zip(tasks, res) if ab is not None]
    validity = float(np.mean(vw)) if vw else float("nan")

    # N0.3 variance: per-topic spread of main adherence across samples
    spreads = [max(v) - min(v) for v in by.values() if len(v) >= 2]
    frac_spread = float(np.mean([sp >= 2 for sp in spreads])) if spreads else float("nan")

    parse_rate = float(np.mean([x is not None for x in raw]))
    np.savez(O / "npov_par_n0.npz", main_sc=main_sc, ceil=ceil, validity=validity,
             frac_spread=frac_spread, spreads=np.array(spreads, float), parse_rate=parse_rate)
    print(log_cost("NPOV", "par_n0_judge", time.time() - t0, device,
                   notes=f"{len(topics)} topics, {len(ctrl)} controls"))

    ceil_gap = (ceil.mean() - main_sc.mean()) if len(ceil) else float("nan")
    print("\n=== S1-NPOV / N0 paragraph-level (gold judge = Qwen-7B) ===")
    print(f"judge parse-rate: {parse_rate:.0%}")
    print(f"N0.1 headroom : OLMo {main_sc.mean():.2f}/5 (frac>=4 {np.mean(main_sc>=4):.0%}, "
          f"frac==5 {np.mean(main_sc>=5):.0%}, frac<=2 {np.mean(main_sc<=2):.0%})  "
          f"vs neutral-control ceiling {ceil.mean():.2f}/5 (gap {ceil_gap:+.2f})")
    print(f"N0.2 validity : gold prefers NEUTRAL over SLANTED control = {validity:.2f}  (need >= 0.90, MASTER)")
    print(f"N0.3 variance : frac topics with sample spread >=2 = {frac_spread:.2f}  (need >= 0.30)")
    g1 = (np.mean(main_sc >= 4) < 0.85) and (ceil_gap >= 0.3)
    g2 = validity >= 0.90
    g3 = frac_spread >= 0.30
    print(f"\nGREEN  N0.1(headroom)={g1}  N0.2(validity)={g2}  N0.3(variance)={g3}  ->  "
          f"{'ALL GREEN: unlock N1 discovery' if (g1 and g2 and g3) else 'STOP: report the failing check'}")


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
