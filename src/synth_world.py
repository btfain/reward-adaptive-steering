"""Stage B0: synthetic positive-control world — GPU phases.

Builds everything the offline learning loop needs, once, on the cluster:

  prompts : type-conditioned short prompts from a HELD-OUT generator
            (types causally upstream -> no labeling error)
  cache   : completion + phi features for every prompt x action
            (25 sparse actions: none + 6 axes x 4 alpha-fractions)
  hidden  : base-model residual state at the steer layer, final prompt token
            (the conditional policy's input features)

Reward is analytic on phi and applied offline (src/synth_learn.py, local), so
one cache serves every (beta, type-dependence, seed) cell of the B0 grid.

Outputs: data/synthetic/{prompts.jsonl, cache.jsonl, hidden.npz}.
"""

import argparse
import json
import time

import numpy as np
import torch
import yaml
from basis_extract import load_basis_config
from models import (
    REPO_ROOT,
    generate_batch,
    load_base,
    load_config,
    log_cost,
    resolve_device,
    resolve_dtype,
)
from proxies import HEDGE, _question_count, _rate
from steer_sanity import measure_ref_norm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = REPO_ROOT / "data" / "synthetic"
BASIS = REPO_ROOT / "basis"

META_REJECT_STARTS = ("here is", "here's", "sure", "certainly", "as an ai", "user:")


def load_synth_config():
    with open(REPO_ROOT / "configs" / "synth.yaml") as f:
        return yaml.safe_load(f)


def phi_features(completion):
    """The reward-defining features. Trivially computable, no learned models."""
    words = max(len(completion.split()), 1)
    return {
        "words": float(len(completion.split())),
        "hedge_per100": _rate(completion, HEDGE),
        "questions_per100": 100.0 * _question_count(completion) / words,
    }


def _clean(text, gcfg, seen):
    t = " ".join(text.strip().split()).strip("\"'“”‘’ ")
    if not (gcfg["min_chars"] <= len(t) <= gcfg["max_chars"]):
        return None
    if t.lower().startswith(META_REJECT_STARTS) or t.lower() in seen:
        return None
    return t


def phase_prompts(cfg, scfg, device):
    """Generate type-conditioned prompts with the held-out generator."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "prompts.jsonl"
    gcfg = scfg["generator"]
    cells = [(t, s) for t in scfg["types"] for s in ("high", "medium", "low")]
    n_total = len(cells) * gcfg["per_cell"]
    if out.exists() and sum(1 for _ in open(out)) == n_total:
        print(f"prompts already complete ({n_total}), skipping generation")
        return

    tok = AutoTokenizer.from_pretrained(gcfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        gcfg["model"], torch_dtype=resolve_dtype(cfg, device)
    ).to(device)
    model.eval()
    gen_cfg = {
        "steer_layer": 0,
        "generation": {
            "max_new_tokens": gcfg["max_new_tokens"],
            "do_sample": True,
            "temperature": gcfg["temperature"],
            "top_p": gcfg["top_p"],
        },
    }

    topics = scfg["topics"]
    rows, seen = [], set()
    for ci, (typ, sal) in enumerate(cells):
        torch.manual_seed(gcfg["seed"] * 1000 + ci)
        kept, attempt = [], 0
        max_attempts = gcfg["per_cell"] * gcfg["attempt_factor"]
        while len(kept) < gcfg["per_cell"] and attempt < max_attempts:
            metas = []
            for _ in range(min(8, max_attempts - attempt)):
                topic = topics[(ci + attempt) % len(topics)]
                metas.append(
                    f"Write one short message (1-3 sentences) that a user might "
                    f"send to an AI assistant, asking for help with {topic}. "
                    f"{typ['persona']} {scfg['salience'][sal]} Write only the "
                    "user's message itself - no quotation marks, no preamble, "
                    "no explanation."
                )
                attempt += 1
            for text in generate_batch(model, tok, metas, gen_cfg):
                t = _clean(text, gcfg, seen)
                if t is not None and len(kept) < gcfg["per_cell"]:
                    kept.append(t)
                    seen.add(t.lower())
        if len(kept) < gcfg["per_cell"]:
            raise RuntimeError(
                f"cell ({typ['name']}, {sal}): only {len(kept)}/{gcfg['per_cell']} "
                f"prompts accepted after {max_attempts} attempts — inspect the "
                "generator output before rerunning (do not lower the filter bar)."
            )
        rows.extend(
            {"id": f"{typ['name']}|{sal}|{i:02d}", "type": typ["name"],
             "salience": sal, "prompt": p}
            for i, p in enumerate(kept)
        )
        print(f"cell {typ['name']}/{sal}: {len(kept)} prompts ({attempt} attempts)")

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"{len(rows)} prompts -> {out}")


def action_list(names, alpha_fracs):
    acts = [{"id": "none", "axis": None, "frac": 0.0}]
    for n in names:
        for f in alpha_fracs:
            acts.append({"id": f"{n}{f:+.1f}", "axis": n, "frac": f})
    return acts


def _complete_actions(path, n_prompts):
    """Resume support: keep only actions with all prompts cached; drop partials."""
    if not path.exists():
        return set()
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(r)
    complete = {a for a, rs in by.items() if len(rs) == n_prompts}
    if len(complete) != len(by):
        path.rename(path.with_suffix(".jsonl.partial.bak"))
        with open(path, "w") as f:
            for r in rows:
                if r["action"] in complete:
                    f.write(json.dumps(r) + "\n")
        print(f"dropped {len(by) - len(complete)} partial action(s) on resume")
    return complete


def phase_cache(cfg, scfg, device, model, tok):
    """Cache completion + phi for every prompt x action."""
    prompts = [json.loads(l) for l in open(DATA / "prompts.jsonl")]
    layer = cfg["steer_layer"]
    names = [a["name"] for a in load_basis_config()["axes"]]
    axes = np.load(BASIS / "axes.npz")
    ref = measure_ref_norm(model, tok, [p["prompt"] for p in prompts[:8]], layer)
    print(f"ref norm {ref:.1f}, axes: {names}")

    ccfg = scfg["cache"]
    gen_cfg = {
        "steer_layer": layer,
        "generation": {
            "max_new_tokens": ccfg["max_new_tokens"],
            "do_sample": True,
            "temperature": ccfg["temperature"],
            "top_p": ccfg["top_p"],
        },
    }
    acts = action_list(names, scfg["actions"]["alpha_fracs"])
    path = DATA / "cache.jsonl"
    done = _complete_actions(path, len(prompts))
    bs = ccfg["batch_size"]

    with open(path, "a") as f:
        for act in acts:
            if act["id"] in done:
                continue
            vec = (
                torch.tensor(axes[f"{act['axis']}|{layer}"])
                if act["axis"] is not None
                else None
            )
            # CRN convention (same as Stage B eval): one RNG stream per action,
            # prompts consumed in fixed order, so streams are matched across actions.
            torch.manual_seed(ccfg["seed"])
            t0 = time.time()
            for i in range(0, len(prompts), bs):
                batch = prompts[i : i + bs]
                comps = generate_batch(
                    model, tok, [p["prompt"] for p in batch], gen_cfg,
                    vector=vec, alpha=act["frac"] * ref if vec is not None else 0.0,
                )
                for p, c in zip(batch, comps):
                    f.write(json.dumps(
                        {"pid": p["id"], "action": act["id"], "axis": act["axis"],
                         "frac": act["frac"], "completion": c,
                         "phi": phi_features(c)}
                    ) + "\n")
                f.flush()
            print(f"action {act['id']}: {len(prompts)} completions "
                  f"({time.time() - t0:.0f}s)")
    print(f"cache -> {path}")


@torch.no_grad()
def phase_hidden(cfg, scfg, device, model, tok):
    """Base-model residual state at the steer layer, final prompt token.

    These are the conditional policy's input features (same convention Stage C
    will use on real prompts), cached so policy training runs without a GPU.
    """
    prompts = [json.loads(l) for l in open(DATA / "prompts.jsonl")]
    layer = cfg["steer_layer"]
    acts = {}

    def hook(_module, _inputs, output):
        acts["h"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[layer].register_forward_hook(hook)
    states = []
    try:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        for i in range(0, len(prompts), 16):
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    add_generation_prompt=True, tokenize=False,
                )
                for p in prompts[i : i + 16]
            ]
            enc = tok(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            model(**enc)
            # left padding -> position -1 is the true final token for every row
            states.append(acts["h"][:, -1, :].float().cpu())
    finally:
        handle.remove()
    mat = torch.cat(states).numpy()
    np.savez(
        DATA / "hidden.npz",
        ids=np.array([p["id"] for p in prompts]),
        states=mat.astype(np.float32),
        layer=layer,
    )
    print(f"hidden states {mat.shape} (layer {layer}) -> {DATA / 'hidden.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["prompts", "cache", "hidden", "all"])
    args = ap.parse_args()
    cfg = load_config()
    scfg = load_synth_config()
    device = resolve_device(cfg)

    if args.phase in ("prompts", "all"):
        t0 = time.time()
        phase_prompts(cfg, scfg, device)
        print(log_cost("B0", "synth_prompts", time.time() - t0, device))
    if args.phase in ("cache", "hidden", "all"):
        model, tok = load_base(cfg, device)
        if args.phase in ("cache", "all"):
            t0 = time.time()
            phase_cache(cfg, scfg, device, model, tok)
            print(log_cost("B0", "synth_cache", time.time() - t0, device))
        if args.phase in ("hidden", "all"):
            t0 = time.time()
            phase_hidden(cfg, scfg, device, model, tok)
            print(log_cost("B0", "synth_hidden", time.time() - t0, device))


if __name__ == "__main__":
    main()
