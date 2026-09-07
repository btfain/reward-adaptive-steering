"""Synthetic notation positive-control — generate + score.

  gen    : for each typed prompt, generate a reply under NULL, each per-type constraint BUNDLE, and the
           FULL_RUBRIC baseline (all conditional rules in one system prompt). Single-turn. (GPU)
  score  : deterministic constraint satisfaction of the PROMPT-TYPE's rules -> (n_prompts x n_moves) matrix.
           No judge, no GPU. -> synth.npz

  python src/synth_swing.py --phase gen   --base-config configs/base_7b.yaml --config configs/synth_notation_v1.yaml
  python src/synth_swing.py --phase score --config configs/synth_notation_v1.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_swing import _gen_multiturn
import synth_scorer as S


def _moves(cfg):
    """[null] + per-type bundles + full_rubric. Each: {id, name, system}."""
    mv = [{"id": 0, "name": "null", "system": ""}]
    for k, t in enumerate(S.types(cfg), start=1):
        mv.append({"id": k, "name": t, "system": S.move_bundle(cfg, t)})
    mv.append({"id": len(mv), "name": "full_rubric", "system": S.full_rubric(cfg)})
    return mv


def _out(cfg):
    d = REPO_ROOT / "results" / cfg["tag"]; d.mkdir(parents=True, exist_ok=True); return d


def phase_gen(base_cfg, cfg, model, tok):
    moves = _moves(cfg); gen = cfg["gen"]; O = _out(cfg)
    P = S.prompts(cfg)
    json.dump(P, open(O / "prompts.json", "w"))
    print(f"prompts: {len(P)} ({cfg['data']['n_per_type']}/type x {len(S.types(cfg))} types)", flush=True)
    B = 16
    with open(O / "gen.jsonl", "w") as f:
        for mv in moves:
            sysmsg = mv["system"] or None
            for rep in range(gen["m_samples"]):
                for s in range(0, len(P), B):
                    chunk = P[s:s + B]
                    convs = [[{"role": "user", "content": x["prompt"]}] for x in chunk]
                    outs = _gen_multiturn(model, tok, convs, gen, sysmsg)
                    for x, a in zip(chunk, outs):
                        f.write(json.dumps({"ci": x["ci"], "move": mv["id"], "s": rep, "text": a}) + "\n")
                f.flush()
            print(f"  gen move {mv['id']} ({mv['name']}) done", flush=True)


def phase_score(cfg):
    O = _out(cfg); moves = _moves(cfg); P = {x["ci"]: x for x in json.load(open(O / "prompts.json"))}
    id2col = {m["id"]: k for k, m in enumerate(moves)}
    n, K = len(P), len(moves)
    Ssum = np.zeros((n, K)); cnt = np.zeros((n, K))
    for l in open(O / "gen.jsonl"):
        r = json.loads(l); typ = P[r["ci"]]["type"]
        sat = S.score(cfg, r["text"], typ)
        col = id2col[r["move"]]; Ssum[r["ci"], col] += sat; cnt[r["ci"], col] += 1
    M = np.where(cnt > 0, Ssum / np.maximum(cnt, 1), np.nan)
    types_per = np.array([P[i]["type"] for i in range(n)], dtype=object)
    ctx = np.array([P[i]["prompt"] for i in range(n)], dtype=object)
    np.savez(O / "synth.npz", M=M, cnt=cnt, contexts=ctx, prompt_types=types_per,
             move_names=np.array([m["name"] for m in moves], dtype=object))
    print(f"score -> {O/'synth.npz'}  (M {M.shape}, complete {int((~np.isnan(M).any(1)).sum())}/{n})")
    print("mean satisfaction by move: " +
          ", ".join(f"{m['name']} {np.nanmean(M[:, id2col[m['id']]]):.2f}" for m in moves))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--tag", default=None, help="override cfg tag (e.g. per base model)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    if args.tag:
        cfg["tag"] = args.tag
    if args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg); t0 = time.time()
        model, tok = load_base(base_cfg, device)
        phase_gen(base_cfg, cfg, model, tok)
        print(log_cost("SYN", "gen", time.time() - t0, device, notes="synthetic notation, single-turn"))
    else:
        phase_score(cfg)


if __name__ == "__main__":
    main()
