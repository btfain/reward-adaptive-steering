"""Stage B0: synthetic positive-control world — GPU phases (v2 manifestation dial).

Builds everything the offline learning loop needs, once, on the cluster:

  prompts : type-conditioned short prompts from a HELD-OUT generator.
            Types are preferences over RESPONSE behavior; the manifestation
            dial controls how (whether) the prompt reveals them:
              explicit    - stated outright
              situational - a situation in which the behavior is what a good
                            response provides (recipes piloted on 1.5B locally)
              none        - one shared neutral pool, types assigned at random,
                            so recoverability is exactly zero by construction
            Hard audit gates run BEFORE the file is written.
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
MANIFESTATIONS = ("explicit", "situational", "none")


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


# ---------------------------------------------------------------- prompts ----

def build_meta(recipe, clause):
    return (
        "You are writing test data: exactly one message that a USER sends to an "
        "AI assistant. You write only the user's side. Do not write the "
        "assistant's reply, do not answer or fulfill the request yourself, and "
        "do not add any preamble.\n\n"
        f"Scenario: {recipe}\n\n"
        "Now write the user's message (1-3 sentences), in the user's "
        f"first-person voice, addressed to the assistant. {clause} Output only "
        "the message text."
    )


def _clean(text, gcfg, seen, reject=()):
    t = " ".join(text.strip().split()).strip("\"'“”‘’ ")
    low = t.lower()
    if not (gcfg["min_chars"] <= len(t) <= gcfg["max_chars"]):
        return None
    if low.startswith(META_REJECT_STARTS) or low in seen:
        return None
    if any(m in low for m in reject):
        return None
    # mojibake / wrong-script guard (curly quotes and dashes stay allowed)
    if any(
        ord(c) > 0x2500 or (0x0370 <= ord(c) <= 0x1FFF and c not in "‘’“”")
        for c in t
    ):
        return None
    return t


def _fill_cell(model, tok, gen_cfg, gcfg, meta_fn, n_needed, seen, reject, seed):
    torch.manual_seed(seed)
    kept, attempt = [], 0
    max_attempts = n_needed * gcfg["attempt_factor"]
    while len(kept) < n_needed and attempt < max_attempts:
        metas = [meta_fn(attempt + j) for j in range(min(8, max_attempts - attempt))]
        attempt += len(metas)
        for text in generate_batch(model, tok, metas, gen_cfg):
            t = _clean(text, gcfg, seen, reject)
            if t is not None and len(kept) < n_needed:
                kept.append(t)
                seen.add(t.lower())
    return kept, attempt


def _audit_prompts(rows, scfg):
    """Print the prompt-mirroring table; hard-gate on leaks and on the
    inquiring/proceeding contrast. Raises before anything is written."""
    names = [t["name"] for t in scfg["types"]]
    rej = {t["name"]: [m.lower() for m in t["reject"]] for t in scfg["types"]}
    words = {}
    print(f"\n{'type':11s} {'manif':12s} {'n':>3s} {'words':>7s} {'hedge/100':>10s} {'?-marks':>8s}")
    for m in MANIFESTATIONS:
        for tn in names:
            sel = [r["prompt"] for r in rows if r["type"] == tn and r["manifestation"] == m]
            words[(tn, m)] = float(np.mean([len(p.split()) for p in sel]))
            print(f"{tn:11s} {m:12s} {len(sel):3d} {words[(tn, m)]:7.1f} "
                  f"{np.mean([_rate(p, HEDGE) for p in sel]):10.2f} "
                  f"{np.mean([p.count('?') for p in sel]):8.2f}")

    problems = []
    leaks = [
        r["id"] for r in rows
        if r["manifestation"] == "situational"
        and any(k in r["prompt"].lower() for k in rej[r["type"]])
    ]
    if leaks:
        problems.append(f"reject-lexicon hits in situational cells: {leaks[:8]}")
    ratio = words[("proceeding", "situational")] / words[("inquiring", "situational")]
    floor = scfg["audit"]["proceed_inquire_word_ratio"]
    print(f"\nproceeding/inquiring situational word ratio: {ratio:.2f} (floor {floor})")
    if ratio < floor:
        problems.append(
            f"proceeding/inquiring word ratio {ratio:.2f} < {floor}: proceeding "
            "prompts are not carrying the specifics that inquiring prompts omit"
        )
    if problems:
        raise RuntimeError("prompt audit FAILED — nothing written: " + "; ".join(problems))


def phase_prompts(cfg, scfg, device):
    """Generate type-conditioned prompts with the held-out generator."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "prompts.jsonl"
    gcfg = scfg["generator"]
    per_cell = gcfg["per_cell"]
    n_total = len(scfg["types"]) * len(MANIFESTATIONS) * per_cell
    if out.exists() and sum(1 for _ in open(out)) == n_total:
        first = json.loads(open(out).readline())
        if "manifestation" in first:
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

    cells = [(t, m) for m in ("explicit", "situational") for t in scfg["types"]]
    for ci, (typ, manif) in enumerate(cells):
        def meta_fn(i, ci=ci, typ=typ, manif=manif):
            topic = topics[(ci + i) % len(topics)]
            if manif == "situational":
                recipe = typ["situational"].format(topic=topic)
                clause = "The message reflects the scenario naturally."
            else:
                recipe = f"A user needs help with {topic}."
                clause = f"The user states outright, in their own words, {typ['explicit']}."
            return build_meta(recipe, clause)

        reject = tuple(m.lower() for m in typ["reject"]) if manif == "situational" else ()
        kept, attempts = _fill_cell(
            model, tok, gen_cfg, gcfg, meta_fn, per_cell, seen, reject,
            seed=gcfg["seed"] * 1000 + ci,
        )
        if len(kept) < per_cell:
            raise RuntimeError(
                f"cell ({typ['name']}, {manif}): only {len(kept)}/{per_cell} prompts "
                f"accepted after {attempts} attempts — inspect the generator output "
                "before rerunning (do not lower the filter bar)."
            )
        rows.extend(
            {"id": f"{typ['name']}|{manif}|{i:02d}", "type": typ["name"],
             "manifestation": manif, "prompt": p}
            for i, p in enumerate(kept)
        )
        print(f"cell {typ['name']}/{manif}: {len(kept)} prompts ({attempts} attempts)")

    # Shared none pool: one neutral distribution, types assigned at random, so
    # MI(type; prompt) = 0 exactly — the floor cell is a built-in negative control.
    n_none = per_cell * len(scfg["types"])
    def meta_none(i):
        return build_meta(
            scfg["none_pool"]["recipe"].format(topic=topics[i % len(topics)]),
            scfg["none_pool"]["clause"],
        )
    kept, attempts = _fill_cell(
        model, tok, gen_cfg, gcfg, meta_none, n_none, seen, (),
        seed=gcfg["seed"] * 1000 + 999,
    )
    if len(kept) < n_none:
        raise RuntimeError(
            f"none pool: only {len(kept)}/{n_none} prompts accepted after "
            f"{attempts} attempts — inspect the generator output before rerunning."
        )
    rng = np.random.default_rng(gcfg["seed"])
    assigned = rng.permutation(np.repeat([t["name"] for t in scfg["types"]], per_cell))
    counters = {t["name"]: 0 for t in scfg["types"]}
    for p, tn in zip(kept, assigned):
        rows.append({"id": f"{tn}|none|{counters[tn]:02d}", "type": str(tn),
                     "manifestation": "none", "prompt": p})
        counters[tn] += 1
    print(f"none pool: {len(kept)} prompts ({attempts} attempts), types randomized")

    _audit_prompts(rows, scfg)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"{len(rows)} prompts -> {out}")


# ------------------------------------------------------------------ cache ----

def _load_prompts():
    prompts = [json.loads(l) for l in open(DATA / "prompts.jsonl")]
    if "manifestation" not in prompts[0]:
        raise RuntimeError(
            "data/synthetic/prompts.jsonl uses the retired salience-dial schema — "
            "regenerate with --phase prompts before caching."
        )
    return prompts


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
    prompts = _load_prompts()
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
    prompts = _load_prompts()
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
