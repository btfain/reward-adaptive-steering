"""M1b — build the multi-turn swing matrix on logged chat.

  gen    : for each logged context (u1,a1,u2), regenerate the assistant reply a2 under NULL + each move
           (move injected as a SYSTEM message), m samples each. Saves text + the logged a2 as reference.
  judge  : Prometheus-2 scores every a2 for context-appropriateness under our rubric (+ logged a2 as the
           score-5 reference). Builds the (n_contexts x n_moves) mean-score matrix -> swing.npz.

Two phases so the 7B base and the 7B judge never sit on the GPU together. Reward = judge, NOT the RM.

  python src/mt_swing.py --phase gen   --base-config configs/base_7b.yaml --config configs/mt_swing_wildchat_v1.yaml
  python src/mt_swing.py --phase judge --config configs/mt_swing_wildchat_v1.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device
from mt_judge import load_judge, load_rubric, score_batch


def _cfg(p):
    return yaml.safe_load(open(REPO_ROOT / p))


def _moves(c):
    m = yaml.safe_load(open(REPO_ROOT / c["moves_config"]))["moves"]
    return m  # list of {id,name,trigger,system}


def _out(c):
    d = REPO_ROOT / "results" / c["tag"]; d.mkdir(parents=True, exist_ok=True); return d


@torch.no_grad()
def _gen_multiturn(model, tok, convs, gen, system):
    """convs: list of message-lists [u1,a1,u2]. Returns generated a2 text for each, under `system`."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(([{"role": "system", "content": system}] if system else []) + conv,
                                     add_generation_prompt=True, tokenize=False) for conv in convs]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=gen["max_new_tokens"], do_sample=True,
                         temperature=gen["temperature"], top_p=gen["top_p"], pad_token_id=tok.eos_token_id)
    w = enc["input_ids"].shape[1]
    return [tok.decode(r[w:], skip_special_tokens=True) for r in out]


def _contexts(c):
    """Stream a multi-turn chat set; keep the first (u1,a1,u2,a2) with a substantive u2."""
    from datasets import load_dataset
    d = c["data"]
    ds = load_dataset(d["dataset"], split=d["split"], streaming=True).shuffle(d["seed"], buffer_size=10000)
    field = d["conversation_field"]; got = []
    for ex in ds:
        conv = ex.get(field)
        if not conv or len(conv) < 4:
            continue
        if d.get("english_only") and ex.get("language") not in (None, "English"):
            continue
        if [t.get("role") for t in conv[:4]] != ["user", "assistant", "user", "assistant"]:
            continue
        u1, a1, u2, a2 = (conv[i]["content"] for i in range(4))
        if len(u2) < d["min_u2_chars"] or len(a1) < d["min_a1_chars"]:
            continue
        if len(u1) + len(a1) + len(u2) > d["max_ctx_chars"]:
            continue
        got.append({"ci": len(got),
                    "ctx": [{"role": "user", "content": u1}, {"role": "assistant", "content": a1},
                            {"role": "user", "content": u2}],
                    "reference": a2})
        if len(got) >= d["n_contexts"]:
            break
    return got


def phase_gen(base_cfg, c, model, tok):
    moves = _moves(c); gen = c["gen"]; O = _out(c)
    ctxs = _contexts(c)
    json.dump(ctxs, open(O / "contexts.json", "w"))
    print(f"contexts: {len(ctxs)}", flush=True)
    B = 16
    with open(O / "gen.jsonl", "w") as f:
        for mv in moves:
            sysmsg = mv["system"] or None
            for rep in range(gen["m_samples"]):
                for s in range(0, len(ctxs), B):
                    chunk = ctxs[s:s + B]
                    outs = _gen_multiturn(model, tok, [x["ctx"] for x in chunk], gen, sysmsg)
                    for x, a2 in zip(chunk, outs):
                        if a2.strip():
                            f.write(json.dumps({"ci": x["ci"], "move": mv["id"], "s": rep, "text": a2}) + "\n")
                f.flush()
            print(f"  gen move {mv['id']} ({mv['name']}) done", flush=True)


def phase_judge(c, shard):
    """RESUMABLE + SHARDABLE judging: score gen items into scores[_shard_i].jsonl, skipping ones already
    done, appending incrementally so a timeout never loses progress. --shard i/N partitions the items."""
    O = _out(c); rubric = load_rubric(c["moves_config"])
    by = {x["ci"]: x for x in json.load(open(O / "contexts.json"))}
    gens = [json.loads(l) for l in open(O / "gen.jsonl")]
    if shard is not None:
        i, N = shard; gens = [g for k, g in enumerate(gens) if k % N == i]
    fp = O / (f"scores_shard_{shard[0]}.jsonl" if shard is not None else "scores.jsonl")
    done = set()
    if fp.exists():
        for l in open(fp):
            r = json.loads(l); done.add((r["ci"], r["move"], r["s"]))
    pending = [g for g in gens if (g["ci"], g["move"], g["s"]) not in done]
    print(f"judge{'' if shard is None else f' shard {shard[0]}/{shard[1]}'}: {len(pending)} pending "
          f"({len(done)} already scored)", flush=True)
    if not pending:
        return
    device = resolve_device(load_config("configs/base_7b.yaml")); t0 = time.time()
    mdl, tok = load_judge(c["judge"]["model"], device); ref_on = c["judge"]["use_reference"]
    CH = 40
    with open(fp, "a") as f:
        for s in range(0, len(pending), CH):
            chunk = pending[s:s + CH]
            items = [(by[g["ci"]]["ctx"], g["text"], (by[g["ci"]]["reference"] if ref_on else None)) for g in chunk]
            sc = score_batch(mdl, tok, items, rubric, c["judge"]["max_new_tokens"])
            for g, x in zip(chunk, sc):
                f.write(json.dumps({"ci": g["ci"], "move": g["move"], "s": g["s"],
                                    "score": x, "len": len(g["text"].split())}) + "\n")
            f.flush()
            print(f"  judged {min(s+CH,len(pending))}/{len(pending)}", flush=True)
    print(log_cost("MT", "judge", time.time() - t0, device, notes=f"shard {shard}"))


def phase_assemble(c):
    """Merge scores shard files -> swing.npz (n_contexts x n_moves mean judge score)."""
    O = _out(c); moves = _moves(c); ctxs = json.load(open(O / "contexts.json"))
    id2col = {mv["id"]: k for k, mv in enumerate(moves)}
    files = sorted(O.glob("scores_shard_*.jsonl")) or [O / "scores.jsonl"]
    n, K = len(ctxs), len(moves)
    S = np.zeros((n, K)); cnt = np.zeros((n, K)); scores_flat, lens_flat = [], []
    tot = 0
    for fpath in files:
        for l in open(fpath):
            r = json.loads(l); tot += 1
            if r["score"] is None:
                continue
            col = id2col[r["move"]]; S[r["ci"], col] += r["score"]; cnt[r["ci"], col] += 1
            scores_flat.append(r["score"]); lens_flat.append(r["len"])
    M = np.where(cnt > 0, S / np.maximum(cnt, 1), np.nan)
    ctx_str = ["\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                         for m in x["ctx"]) for x in ctxs]
    np.savez(O / "swing.npz", M=M, cnt=cnt, contexts=np.array(ctx_str, dtype=object),
             move_names=np.array([mv["name"] for mv in moves], dtype=object),
             scores_flat=np.array(scores_flat, float), lens_flat=np.array(lens_flat, float))
    parsed = int(cnt.sum())
    print(f"assemble -> {O/'swing.npz'}  (M {M.shape}, parsed {parsed}/{tot}, "
          f"contexts with all {K} moves: {int((~np.isnan(M).any(1)).sum())}/{n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gen", "judge", "assemble"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--shard", default=None, help="i/N to shard the judge phase")
    args = ap.parse_args()
    c = _cfg(args.config)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    if args.phase == "gen":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg); t0 = time.time()
        model, tok = load_base(base_cfg, device)
        phase_gen(base_cfg, c, model, tok)
        print(log_cost("MT", "gen", time.time() - t0, device, notes="multi-turn a2 under moves"))
    elif args.phase == "judge":
        phase_judge(c, shard)
    else:
        phase_assemble(c)


if __name__ == "__main__":
    main()
