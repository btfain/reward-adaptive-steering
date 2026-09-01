"""P1(i) verify — LIKELIHOOD version (interpretability + reward-free). The reward smoke test asks 'does
the move produce higher-reward output?'. This asks the subtly different 'does the move make the SPECIFIC
high-reward completion more likely than the low-reward one?' — i.e. does it explain the contrast it was
derived from. The margin

    shift(x,m) = [logP(y_high|x,m) − logP(y_low|x,m)] − [logP(y_high|x) − logP(y_low|x)]      (mean-token)

is the model's own implicit PREFERENCE shift (DPO-style) induced by move m — reward-free (uses only the
preference pair, no RM) and cheap (teacher-forced, one forward pass, no autoregressive generation). y_high
/ y_low are the max-/min-RM base completions of the source prompt (from the pool). Sharded like the reward
smoke; assemble COMPARES the two filters (do reward- and preference-validation agree?).

    python src/likelihood_smoke.py --phase test --shard 0/4 --base-config configs/base_7b.yaml
    python src/likelihood_smoke.py --phase assemble
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models import REPO_ROOT, load_base, load_config, log_cost, resolve_device

BASIS = REPO_ROOT / "basis"


def _read(path):
    return [l.strip() for l in open(REPO_ROOT / path) if l.strip() and not l.startswith("#")]


def _out(tag):
    d = REPO_ROOT / "results" / f"likelihood_smoke_{tag}"; d.mkdir(parents=True, exist_ok=True); return d


@torch.no_grad()
def _mean_logprob(model, tok, prompt, completion, system, device):
    """Mean per-token logP(completion | prompt[, system]) — length-normalized to avoid length bias."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    comp = tok(completion, return_tensors="pt", add_special_tokens=False)
    ids = torch.cat([prefix["input_ids"], comp["input_ids"]], dim=1).to(device)
    n = prefix["input_ids"].shape[1]
    logits = model(ids).logits[0, n - 1:-1, :]                      # positions predicting the completion tokens
    cids = comp["input_ids"][0].to(device)
    lp = F.log_softmax(logits.float(), dim=-1).gather(1, cids[:, None]).squeeze(1)
    return float(lp.mean())


def phase_test(args, base_cfg, device):
    cands = _read(args.candidates)
    prov = json.load(open(REPO_ROOT / args.provenance))
    # per source prompt: y_high (max RM) and y_low (min RM) base completions
    by = defaultdict(list)
    for l in open(REPO_ROOT / args.pool):
        r = json.loads(l)
        if r["completion"].strip():
            by[r["prompt"]].append((r["rm"], r["completion"]))
    hilo = {}
    for p, v in by.items():
        if len(v) >= 2:
            v = sorted(v)
            hilo[p] = (v[-1][1], v[0][1])                           # (y_high, y_low)
    i, N = (int(x) for x in args.shard.split("/"))
    mine = [(gi, c) for gi, c in enumerate(cands) if gi % N == i]

    model, tok = load_base(base_cfg, device)
    base_margin = {}                                               # cache logP(hi|x)-logP(lo|x) per source prompt
    def margin(prompt, hi, lo, system):
        return _mean_logprob(model, tok, prompt, hi, system, device) - _mean_logprob(model, tok, prompt, lo, system, device)

    out = _out(args.tag) / f"shard_{i}.jsonl"
    with open(out, "w") as f:
        for n, (gi, cand) in enumerate(mine):
            srcs = [p for p in prov.get(cand, []) if p in hilo]
            if not srcs:
                f.write(json.dumps({"candidate": cand, "shift": None}) + "\n"); continue
            shifts = []
            for p in srcs:
                hi, lo = hilo[p]
                if p not in base_margin:
                    base_margin[p] = margin(p, hi, lo, None)
                shifts.append(margin(p, hi, lo, cand) - base_margin[p])
            f.write(json.dumps({"candidate": cand, "shift": float(np.mean(shifts))}) + "\n"); f.flush()
            if (n + 1) % 25 == 0:
                print(f"  shard {i}: {n+1}/{len(mine)}", flush=True)
    print(f"likelihood shard {i} -> {out}", flush=True)


def phase_assemble(args):
    L = {json.loads(l)["candidate"]: json.loads(l)["shift"]
         for f in sorted(_out(args.tag).glob("shard_*.jsonl")) for l in open(f)}
    # reward swings from the reward smoke test, for the comparison
    R = {}
    rdir = REPO_ROOT / "results" / f"smoke_candidates_{args.tag}"
    for f in sorted(rdir.glob("shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l); R[r["candidate"]] = r["swing"]
    common = [c for c in L if L[c] is not None and R.get(c) is not None]
    ls = np.array([L[c] for c in common]); rs = np.array([R[c] for c in common])
    corr = float(np.corrcoef(ls, rs)[0, 1]) if len(common) > 2 else float("nan")
    lk = ls > 0; rk = rs > args.threshold
    both = int((lk & rk).sum()); ronly = int((~lk & rk).sum()); lonly = int((lk & ~rk).sum()); neither = int((~lk & ~rk).sum())

    kept = [c for c in common if L[c] > 0]
    (REPO_ROOT / args.out).write_text(
        "# candidates whose move increases the model's PREFERENCE for the high-reward completion (shift>0)\n"
        + "\n".join(kept) + "\n")
    rows = [f"# P1(i) LIKELIHOOD smoke — does the move explain the contrast? (reward-free, interpretability)\n",
            f"{len(common)} candidates with both signals. Preference-shift kept {int(lk.sum())} ({lk.mean()*100:.0f}%).\n",
            f"## Reward vs preference validation — do they agree?",
            f"- correlation(reward swing, preference shift) = **{corr:+.3f}**",
            f"- both pass: {both} | reward-only: {ronly} | preference-only: {lonly} | neither: {neither}",
            f"- ⇒ reward∧preference agree on {(both+neither)/len(common)*100:.0f}% of candidates; "
            f"{ronly} raise reward WITHOUT explaining the contrast (works for another reason), "
            f"{lonly} explain the contrast WITHOUT raising on-policy reward.",
            "", "## Reading",
            ("- high correlation ⇒ the two validations largely agree; preference shift is a cheap reward-free "
             "proxy for the reward smoke test (and confirms the moves work for the reason the generator guessed)."
             if corr > 0.3 else
             "- LOW correlation ⇒ reward and mechanism diverge — many moves raise reward for reasons OTHER than "
             "the guessed contrast (or vice versa); the two filters are genuinely different views, worth reporting.")]
    BASIS.mkdir(exist_ok=True)
    (BASIS / f"s1_likelihood_smoke_{args.tag}_report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\npreference-verified {len(kept)}/{len(common)} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["test", "assemble"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--candidates", default="configs/candidates_raw_7b.txt")
    ap.add_argument("--provenance", default="configs/candidates_raw_7b.provenance.json")
    ap.add_argument("--pool", default="results/prompt_basis_large_7b/pool.jsonl")
    ap.add_argument("--threshold", type=float, default=-0.2, help="reward-swing threshold for the comparison")
    ap.add_argument("--out", default="configs/candidates_pref_verified_7b.txt")
    args = ap.parse_args()
    if args.phase == "test":
        base_cfg = load_config(args.base_config); device = resolve_device(base_cfg)
        t0 = time.time()
        phase_test(args, base_cfg, device)
        print(log_cost("S1", "likelihood_smoke", time.time() - t0, device, notes="preference-shift verify (teacher-forced)"))
    else:
        phase_assemble(args)


if __name__ == "__main__":
    main()
