"""Coding-mentor C0.5 — the RIGHT baseline is the SYSTEM PROMPT (mentor rubric as instructions), not
best-of-n. Reusing C0's null + seed-move generations, add two arms and judge the reframed questions:
  * sys              — mentor rubric as a system prompt (the practitioner baseline)
  * sys+<move>       — system prompt PLUS a per-context move (the ADDITIVE mode)
Comparisons (Prometheus + mentor rubric, pairwise):
  * sys vs null            — does prompting fix the base at all? (headroom of the prompt / semantic self-apply)
  * move vs sys            — a single move INSTEAD OF the system prompt
  * (sys+move) vs sys      — a single move IN ADDITION TO the system prompt (the new lever)

  python src/mentor_c05.py --phase gen   --base-config configs/base_7b.yaml --config configs/mentor_moves_v1.yaml
  python src/mentor_c05.py --phase judge --config configs/mentor_moves_v1.yaml
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_judge import load_judge, prefer_batch, score_batch
from mt_swing import _gen_multiturn

NPAIR, BOTH = 1, True                         # keep judge cost modest for the de-risk


def _cfg(p):
    return yaml.safe_load(open(REPO_ROOT / p))


def _out():
    return REPO_ROOT / "results" / "mentor_c0"    # reuse C0's dir (prompts.json + gen.jsonl)


def _extra_arms(c):
    """New arms to generate: 'sys' and 'sys+<move>' for each non-null move."""
    sp = c["system_prompt"]
    arms = [("sys", sp)]
    for m in c["moves"]:
        if m["id"] != 0:
            arms.append((f"sys+{m['name']}", sp + "\n\n" + m["system"]))
    return arms


def phase_gen(base_cfg, c, model, tok):
    O = _out(); gen = c["gen"]
    P = json.load(open(O / "prompts.json"))
    B = 16
    with open(O / "gen_sys.jsonl", "w") as f:
        for arm, sysmsg in _extra_arms(c):
            for rep in range(gen["m_samples"]):
                for s in range(0, len(P), B):
                    chunk = P[s:s + B]
                    convs = [[{"role": "user", "content": x["prompt"]}] for x in chunk]
                    for x, a in zip(chunk, _gen_multiturn(model, tok, convs, gen, sysmsg)):
                        if a.strip():
                            f.write(json.dumps({"ci": x["ci"], "arm": arm, "s": rep, "text": a}) + "\n")
                f.flush()
            print(f"  gen arm {arm} done", flush=True)


def _winrate(mdl, tok, rubric, prompts, A_by, B_by, mnt):
    """Pairwise win-rate of responses in A_by[ci] over B_by[ci] (both orders), averaged over prompts."""
    tasks, items = [], []
    for ci in prompts:
        A, Bn = A_by.get(ci, []), B_by.get(ci, [])
        ctx = [{"role": "user", "content": prompts[ci]}]
        for p in range(min(NPAIR, len(A), len(Bn))):
            for tag, ra, rb in ([("A", A[p], Bn[p]), ("B", Bn[p], A[p])] if BOTH else [("A", A[p], Bn[p])]):
                tasks.append((ci, tag)); items.append((ctx, ra, rb, None))
    res = prefer_batch(mdl, tok, items, rubric, mnt)
    wins = defaultdict(list)
    for (ci, tag), ab in zip(tasks, res):
        if ab is not None:
            wins[ci].append(int((tag == "A" and ab == "A") or (tag == "B" and ab == "B")))
    per = [np.mean(v) for v in wins.values() if v]
    return float(np.mean(per)) if per else float("nan")


def phase_judge(c):
    O = _out(); rubric = c["rubric"]; mnt = c["judge"]["max_new_tokens"]
    prompts = {x["ci"]: x["prompt"] for x in json.load(open(O / "prompts.json"))}
    id2name = {m["id"]: m["name"] for m in c["moves"]}
    # null + move arms from C0's gen.jsonl
    base = defaultdict(list); move = defaultdict(lambda: defaultdict(list))
    for l in open(O / "gen.jsonl"):
        r = json.loads(l)
        if r["move"] == 0:
            base[r["ci"]].append(r["text"])
        else:
            move[id2name[r["move"]]][r["ci"]].append(r["text"])
    # sys + sys+move arms from gen_sys.jsonl
    sysd = defaultdict(list); sysmv = defaultdict(lambda: defaultdict(list))
    for l in open(O / "gen_sys.jsonl"):
        r = json.loads(l)
        if r["arm"] == "sys":
            sysd[r["ci"]].append(r["text"])
        else:
            sysmv[r["arm"][4:]][r["ci"]].append(r["text"])

    device = resolve_device(load_config("configs/base_7b.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device)
    movenames = [m["name"] for m in c["moves"] if m["id"] != 0]

    sys_abs = np.array([s for s in score_batch(
        mdl, tok, [([{"role": "user", "content": prompts[ci]}], t, None) for ci in sorted(sysd) for t in sysd[ci]],
        rubric, mnt) if s is not None], float)
    sys_vs_null = _winrate(mdl, tok, rubric, prompts, sysd, base, mnt)
    move_vs_sys = {nm: _winrate(mdl, tok, rubric, prompts, move[nm], sysd, mnt) for nm in movenames}
    sysmove_vs_sys = {nm: _winrate(mdl, tok, rubric, prompts, sysmv[nm], sysd, mnt) for nm in movenames}

    np.savez(O / "mentor_c05.npz", sys_abs=sys_abs,
             move_names=np.array(movenames, dtype=object),
             move_vs_sys=np.array([move_vs_sys[n] for n in movenames], float),
             sysmove_vs_sys=np.array([sysmove_vs_sys[n] for n in movenames], float),
             sys_vs_null=sys_vs_null)
    print(log_cost("MENTOR", "c05_judge", time.time() - t0, device, notes="sys-prompt baseline + additive"))
    print("\n=== C0.5 (baseline = SYSTEM PROMPT) ===")
    print(f"sys absolute mentor score: mean {sys_abs.mean():.2f}/5  (frac>=4: {np.mean(sys_abs>=4):.0%}, "
          f"frac<=2: {np.mean(sys_abs<=2):.0%})   [C0 null base was 2.50]")
    print(f"sys vs null win-rate: {sys_vs_null:.2f}   (does the system prompt beat no-prompt?)")
    print("INSTEAD-OF  (move vs sys):")
    for n in movenames: print(f"    {n:18s} {move_vs_sys[n]:.2f}")
    print("ADDITIVE    (sys+move vs sys):")
    for n in movenames: print(f"    {n:18s} {sysmove_vs_sys[n]:.2f}")
    print("Green if sys leaves headroom (not ~all >=4) AND some move/additive win-rate clearly >0.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "judge"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    args = ap.parse_args()
    c = _cfg(args.config)
    if args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg); t0 = time.time()
        model, tok = load_base(base_cfg, device)
        phase_gen(base_cfg, c, model, tok)
        print(log_cost("MENTOR", "c05_gen", time.time() - t0, device, notes="sys + additive arms"))
    else:
        phase_judge(c)


if __name__ == "__main__":
    main()
