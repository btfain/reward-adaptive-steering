"""Coding-mentor C0 — headroom + judge sanity, BEFORE the discovery pipeline. On coding-help prompts
(WildChat, single-turn), generate the frozen base reply + a few hand-written seed mentor moves, then judge
with Prometheus + the mentor rubric:
  * ABSOLUTE score of the base reply (is the base default un-pedagogical => is there headroom?)
  * PAIRWISE move-vs-base win-rate (do mentor moves materially beat the base => reward discriminates?)

GREEN => base scores low AND at least one seed move clearly beats it => build the discovery pipeline (C1+).

  python src/mentor_c0.py --phase gen   --base-config configs/base_7b.yaml --config configs/mentor_moves_v1.yaml
  python src/mentor_c0.py --phase judge --config configs/mentor_moves_v1.yaml
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_judge import load_judge, load_rubric, prefer_batch, score_batch
from mt_swing import _gen_multiturn

_CODE = re.compile(r"```|\b(python|javascript|java|c\+\+|golang|rust|函数|function|def |return|error|"
                   r"exception|traceback|compile|syntax|npm|pip|import|sql|regex|array|null pointer|segfault|"
                   r"variable|for loop|recursion|api|json|html|css|bug|debug|code)\b", re.I)


def _cfg(p):
    return yaml.safe_load(open(REPO_ROOT / p))


def _out(c):
    d = REPO_ROOT / "results" / c["tag"]; d.mkdir(parents=True, exist_ok=True); return d


def _coding_prompts(c):
    """Stream WildChat; keep first user turns that look like a coding question."""
    from datasets import load_dataset
    d = c["data"]
    ds = load_dataset(d["dataset"], split=d["split"], streaming=True).shuffle(d["seed"], buffer_size=10000)
    got = []
    for ex in ds:
        conv = ex.get(d["conversation_field"])
        if not conv or conv[0].get("role") != "user":
            continue
        if d.get("english_only") and ex.get("language") not in (None, "English"):
            continue
        u = conv[0]["content"]
        if not (d["min_chars"] <= len(u) <= d["max_chars"]) or len(_CODE.findall(u)) < 2:
            continue
        got.append({"ci": len(got), "prompt": u})
        if len(got) >= d["n_prompts"]:
            break
    return got


def phase_gen(base_cfg, c, model, tok):
    moves = c["moves"]; gen = c["gen"]; O = _out(c)
    P = _coding_prompts(c)
    json.dump(P, open(O / "prompts.json", "w"))
    print(f"coding prompts: {len(P)}", flush=True)
    B = 16
    with open(O / "gen.jsonl", "w") as f:
        for mv in moves:
            sysmsg = mv["system"] or None
            for rep in range(gen["m_samples"]):
                for s in range(0, len(P), B):
                    chunk = P[s:s + B]
                    convs = [[{"role": "user", "content": x["prompt"]}] for x in chunk]
                    for x, a in zip(chunk, _gen_multiturn(model, tok, convs, gen, sysmsg)):
                        if a.strip():
                            f.write(json.dumps({"ci": x["ci"], "move": mv["id"], "s": rep, "text": a}) + "\n")
                f.flush()
            print(f"  gen move {mv['id']} ({mv['name']}) done", flush=True)


def phase_judge(c):
    O = _out(c); moves = c["moves"]; rubric = load_rubric(c["moves_config"]) if "moves_config" in c else c["rubric"]
    P = {x["ci"]: x for x in json.load(open(O / "prompts.json"))}
    by = defaultdict(lambda: defaultdict(list))
    for l in open(O / "gen.jsonl"):
        r = json.loads(l); by[r["ci"]][r["move"]].append(r["text"])
    device = resolve_device(load_config("configs/base_7b.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device); mnt = c["judge"]["max_new_tokens"]
    npair = c["pairwise"]["n_pairs"]; both = c["pairwise"]["both_orders"]

    # (a) absolute mentor score of the BASE replies
    base_items = [([{"role": "user", "content": P[ci]["prompt"]}], t, None)
                  for ci in sorted(by) for t in by[ci].get(0, [])]
    base_scores = score_batch(mdl, tok, base_items, rubric, mnt)

    # (b) pairwise move-vs-base win-rate (batched)
    id2name = {m["id"]: m["name"] for m in moves}
    tasks, items = [], []
    for ci in sorted(by):
        base = by[ci].get(0, [])
        if not base:
            continue
        ctx = [{"role": "user", "content": P[ci]["prompt"]}]
        for mv in moves:
            if mv["id"] == 0:
                continue
            mtx = by[ci].get(mv["id"], [])
            for p in range(min(npair, len(mtx), len(base))):
                for tag, ra, rb in ([("M", mtx[p], base[p]), ("B", base[p], mtx[p])] if both
                                    else [("M", mtx[p], base[p])]):
                    tasks.append((ci, mv["id"], tag)); items.append((ctx, ra, rb, None))
    res = prefer_batch(mdl, tok, items, rubric, mnt)
    wins = defaultdict(list)
    for (ci, mid, tag), ab in zip(tasks, res):
        if ab is not None:
            wins[(ci, mid)].append(int((tag == "M" and ab == "A") or (tag == "B" and ab == "B")))
    n = len(P)
    winrate = {mid: np.mean([np.mean(wins[(ci, mid)]) for ci in range(n) if (ci, mid) in wins])
               for mid in id2name if mid != 0}
    np.savez(O / "mentor_c0.npz", base_abs=np.array(base_scores, float),
             move_names=np.array([id2name[m] for m in sorted(id2name) if m != 0], dtype=object),
             winrates=np.array([winrate[m] for m in sorted(id2name) if m != 0], float))
    ba = np.array([s for s in base_scores if s is not None], float)
    print(log_cost("MENTOR", "c0_judge", time.time() - t0, device, notes=f"{len(P)} coding prompts"))
    print(f"\n=== C0 headroom ===\nbase absolute mentor score: mean {ba.mean():.2f} / 5 "
          f"(frac <=2 'just gives solution': {np.mean(ba<=2):.0%})")
    for m in sorted(id2name):
        if m: print(f"  {id2name[m]:18s} win-rate vs base: {winrate[m]:.2f}")
    print("GREEN if base mean is low (~<=3) AND a move win-rate is clearly >0.5")


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
        print(log_cost("MENTOR", "c0_gen", time.time() - t0, device, notes="base + seed mentor moves"))
    else:
        phase_judge(c)


if __name__ == "__main__":
    main()
