"""Stage A2: headroom diagnostic — does the real RM respond to our action space?

Fixed grid of interventions per prompt, paired within-prompt scoring vs the
no-intervention baseline, on real UltraFeedback prompts. NO learned policy: this
measures the CEILING a conditional policy could exploit, and discriminates
H1 (noise) / H2 (dense composition) / H3 (RM flat to style).

Modalities:
  M1  activation steering (base model, extracted vectors) — 6 axes x 4 alphas
  M2  natural-language imperatives (system message) — base AND large model
  dense  multi-axis M1 combos (H2 probe, base only)

Guards: perplexity under the unsteered model (fluency), paired CRN scoring.
The intervention (vector or imperative) is the ACTION and is NEVER shown to the
RM — the RM always scores (original user prompt, completion), so M1 and M2 are
compared on identical footing.

Runs one model at a time (memory) with the RM resident throughout. The full run
splits into a base job and a large job; the pilot runs both on few prompts x
several seeds for a go/no-go.

Outputs: results/headroom/headroom_log.jsonl (resumable), basis/headroom_pilot.md.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from basis_extract import load_basis_config
from fixed_steer import combined_vector, project_to_cap
from models import (
    REPO_ROOT,
    generate_batch,
    load_base,
    load_config,
    load_rm,
    log_cost,
    resolve_device,
    resolve_dtype,
    rm_score,
)
from nl_control import m2_conditions
from steer_sanity import measure_ref_norm
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = REPO_ROOT / "results" / "headroom"
BASIS = REPO_ROOT / "basis"


def load_hr_config():
    with open(REPO_ROOT / "configs" / "headroom.yaml") as f:
        return yaml.safe_load(f)


def dense_specs(names, hcfg):
    """(name, coeff-vector) list for the H2 probe: Stage B CEM best, Stage A
    prior, and random dense actions (all axes at +/-0.1)."""
    specs = []
    cem = REPO_ROOT / "results" / "fixed_steer" / "cem_state.json"
    if cem.exists():
        act = json.load(open(cem))["best"]["action"]
        specs.append(("cem_best", np.array([act.get(n, 0.0) for n in names])))
    prior = yaml.safe_load(open(REPO_ROOT / "configs" / "fixed_steer.yaml"))
    pm = prior["search"]["init_mean"]
    specs.append(("prior", np.array([pm.get(n, 0.0) for n in names])))
    rng = np.random.default_rng(0)
    for j in range(hcfg["grid"]["dense_n_random"]):
        specs.append((f"rand{j}", 0.1 * rng.choice([-1.0, 1.0], size=len(names))))
    return specs


def build_conditions(hcfg, names, arms):
    """Full condition list for a model, filtered to its allowed arms."""
    conds = [{"id": "none", "arm": "none"}]
    for n in names:
        for f in hcfg["grid"]["alpha_fracs"]:
            conds.append({"id": f"m1:{n}{f:+.1f}", "arm": "m1", "axis": n, "frac": f})
    conds += m2_conditions(hcfg["imperatives"], names)
    for nm, coeffs in dense_specs(names, hcfg):
        conds.append({"id": f"dense:{nm}", "arm": "dense", "coeffs": coeffs})
    return [c for c in conds if c["arm"] in arms]


@torch.no_grad()
def mean_completion_nll(model, tok, prompt, completion):
    """Mean per-token NLL of the completion under the UNSTEERED model, given the
    clean (imperative-free) prompt prefix — the fluency guard. Higher = less fluent."""
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    comp = tok(completion, return_tensors="pt", add_special_tokens=False)
    if comp["input_ids"].shape[1] == 0:
        return float("nan")
    ids = torch.cat([prefix["input_ids"], comp["input_ids"]], dim=1).to(model.device)
    n_prefix = prefix["input_ids"].shape[1]
    logits = model(ids).logits[0].float()
    logp = F.log_softmax(logits[n_prefix - 1 : -1], dim=-1)
    tgt = ids[0, n_prefix:]
    return float(-logp[torch.arange(len(tgt)), tgt].mean().item())


def _complete_keys(path, model_key, n_prompts):
    """Resume: set of (condition_id, seed) fully cached for this model."""
    if not path.exists():
        return set()
    by = {}
    for l in open(path):
        r = json.loads(l)
        if r["model"] == model_key:
            by.setdefault((r["condition"], r["seed"]), 0)
            by[(r["condition"], r["seed"])] += 1
    return {k for k, c in by.items() if c == n_prompts}


def run_model(model, tok, rm, rm_tok, model_key, prompts, conds, hcfg,
              names, axes, ref, layer, seeds, logf, done):
    gcfg = {"steer_layer": layer, "generation": {
        "max_new_tokens": hcfg["generation"]["max_new_tokens"],
        "do_sample": True,
        "temperature": hcfg["generation"]["temperature"],
        "top_p": hcfg["generation"]["top_p"],
    }}
    bs = hcfg["generation"]["batch_size"]
    cap = hcfg["grid"]["cap_combined_norm"]
    for cond in conds:
        vec, alpha, system = None, 0.0, None
        if cond["arm"] == "m1":
            vec = torch.tensor(axes[f"{cond['axis']}|{layer}"])
            alpha = cond["frac"] * ref
        elif cond["arm"] == "m2":
            system = cond["system"]
        elif cond["arm"] == "dense":
            coeffs, _ = project_to_cap(axes, names, layer, cond["coeffs"], cap)
            vec, alpha = combined_vector(axes, names, layer, coeffs), ref
        for seed in seeds:
            if (cond["id"], seed) in done:
                continue
            torch.manual_seed(seed)                       # CRN: reset per (cond, seed)
            t0 = time.time()
            for i in range(0, len(prompts), bs):
                batch = prompts[i : i + bs]
                comps = generate_batch(
                    model, tok, batch, gcfg, system=system, vector=vec, alpha=alpha
                )
                for p, c in zip(batch, comps):
                    logf.write(json.dumps({
                        "model": model_key, "condition": cond["id"], "arm": cond["arm"],
                        "seed": seed, "prompt": p, "completion": c,
                        "rm": rm_score(rm, rm_tok, p, c),
                        "nll": mean_completion_nll(model, tok, p, c),
                        "words": len(c.split()),
                    }) + "\n")
                logf.flush()
            print(f"[{model_key}] {cond['id']} seed {seed}: "
                  f"{len(prompts)} gens ({time.time() - t0:.0f}s)")


def load_named(model_id, cfg, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=resolve_dtype(cfg, device)
    ).to(device)
    model.eval()
    return model, tok


def report_pilot(hcfg, prompts):
    """Go/no-go readout: per-condition paired DeltaRM (fluency-flagged) and the
    per-prompt oracle headroom, per model."""
    rows = [json.loads(l) for l in open(OUT / "headroom_log.jsonl")]
    flag = hcfg["perplexity"]["flag_frac"]
    lines = ["# Stage A2 pilot — go/no-go readout\n",
             f"{len(prompts)} prompts, seeds {hcfg['pilot']['seeds']}. Paired within-prompt "
             "vs `none`; DeltaPPL = mean completion-perplexity inflation over none "
             f"(flag >{flag:.0%}).\n"]
    models = sorted({r["model"] for r in rows})
    for mk in models:
        mr = [r for r in rows if r["model"] == mk]
        none_rm = {(r["prompt"], r["seed"]): r["rm"] for r in mr if r["condition"] == "none"}
        none_nll = {(r["prompt"], r["seed"]): r["nll"] for r in mr if r["condition"] == "none"}
        conds = [c for c in dict.fromkeys(r["condition"] for r in mr) if c != "none"]

        per_cond = {}
        for c in conds:
            cr = [r for r in mr if r["condition"] == c]
            # average seeds within prompt, then aggregate over prompts (honest n)
            by_p = {}
            for r in cr:
                k = (r["prompt"], r["seed"])
                if k in none_rm:
                    by_p.setdefault(r["prompt"], []).append((
                        r["rm"] - none_rm[k],
                        (r["nll"] - none_nll[k]) / max(none_nll[k], 1e-6),
                    ))
            drm = np.array([np.mean([d[0] for d in v]) for v in by_p.values()])
            dpp = np.array([np.mean([d[1] for d in v]) for v in by_p.values()])
            per_cond[c] = {
                "drm": float(drm.mean()),
                "se": float(drm.std() / np.sqrt(max(len(drm), 1))),
                "dppl": float(dpp.mean()),
                "flagged": bool(dpp.mean() > flag),
            }

        lines.append(f"\n## {mk} — per-condition paired ΔRM (sorted; ⚑ = perplexity-flagged)\n")
        lines.append("| condition | ΔRM | ±SE | ΔPPL | |")
        lines.append("|---|---|---|---|---|")
        for c in sorted(per_cond, key=lambda c: -per_cond[c]["drm"]):
            d = per_cond[c]
            lines.append(f"| {c} | {d['drm']:+.3f} | {d['se']:.3f} | "
                         f"{d['dppl']:+.0%} | {'⚑' if d['flagged'] else ''} |")

        # per-prompt oracle headroom over fluency-passing conditions
        ok = {c for c in conds if not per_cond[c]["flagged"]}
        oracle, winners = [], []
        for p in {r["prompt"] for r in mr}:
            for s in hcfg["pilot"]["seeds"]:
                if (p, s) not in none_rm:
                    continue
                cand = [(r["condition"], r["rm"] - none_rm[(p, s)]) for r in mr
                        if r["prompt"] == p and r["seed"] == s and r["condition"] in ok]
                if cand:
                    w, best = max(cand, key=lambda t: t[1])
                    oracle.append(best)
                    winners.append(w)
        best_c = max(per_cond, key=lambda c: per_cond[c]["drm"])
        bd = per_cond[best_c]
        lines.append(
            f"\n- **best mean condition:** {best_c} ΔRM {bd['drm']:+.3f} ± {bd['se']:.3f}"
            f" ({'flagged' if bd['flagged'] else 'fluent'})"
        )
        lines.append(f"- **oracle per-prompt headroom** (fluent conds): "
                     f"{np.mean(oracle):+.3f} (mean best-per-prompt ΔRM)")
        uniq = len(set(winners))
        lines.append(f"- **argmax winners:** {uniq} distinct conditions win across "
                     f"prompts×seeds — {'prompt-dependent' if uniq > 3 else 'concentrated'}")

    lines.append(
        "\n## Reading\nGO if any model shows a fluent condition with ΔRM > ~2·SE and "
        "positive oracle headroom — there is something to optimize. REBRIEF if "
        "everything is flat or only flagged (degradation) conditions move the RM "
        "(early H3 signal); discuss before the full 200-prompt run."
    )
    (BASIS / "headroom_pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport -> {BASIS / 'headroom_pilot.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--model", default="both", choices=["base", "large", "both"])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    hcfg = load_hr_config()
    device = resolve_device(cfg)
    layer = cfg["steer_layer"]
    names = [a["name"] for a in load_basis_config()["axes"]]
    mcfg = hcfg[args.mode]
    data = json.load(open(REPO_ROOT / "data" / "prompts.json"))
    prompts = data["train"][: mcfg["n_prompts"]]
    seeds = mcfg["seeds"]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report_pilot(hcfg, prompts)
        return

    axes = np.load(BASIS / "axes.npz")
    rm, rm_tok = load_rm(cfg, device)
    log_path = OUT / "headroom_log.jsonl"
    model_keys = ["base", "large"] if args.model == "both" else [args.model]

    t0 = time.time()
    with open(log_path, "a") as logf:
        for mk in model_keys:
            conds = build_conditions(hcfg, names, hcfg["models"][mk]["arms"])
            done = _complete_keys(log_path, mk, len(prompts))
            print(f"=== {mk}: {len(conds)} conditions x {len(seeds)} seeds "
                  f"({len(done)} already done) ===")
            if mk == "base":
                model, tok = load_base(cfg, device)
                ref = measure_ref_norm(model, tok, prompts[:8], layer)
                print(f"ref norm {ref:.1f}")
            else:
                model, tok = load_named(hcfg["models"][mk]["model"], cfg, device)
                ref = None
            run_model(model, tok, rm, rm_tok, mk, prompts, conds, hcfg,
                      names, axes, ref, layer, seeds, logf, done)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(log_cost("A2", f"headroom_{args.mode}_{args.model}", time.time() - t0, device))
    if args.mode == "pilot":
        report_pilot(hcfg, prompts)


if __name__ == "__main__":
    main()
