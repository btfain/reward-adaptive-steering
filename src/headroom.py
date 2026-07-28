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
import collections
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
# full run uses separate files: its 200 prompts overlap the pilot's 10, so
# appending to the pilot log would create duplicate (cond, seed, prompt) keys.
FULL_LOG = OUT / "headroom_full_log.jsonl"
FULL_RM2 = OUT / "headroom_full_rm2.jsonl"


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


def _fmt_frac(f):
    """+0.1 / -0.05 / +0.2 — matches the base pilot's id format across bands."""
    return f"{f:+.2f}".rstrip("0").rstrip(".")


def build_conditions(hcfg, names, arms, alpha_fracs):
    """Full condition list for a model, filtered to its allowed arms."""
    conds = [{"id": "none", "arm": "none"}]
    for n in names:
        for f in alpha_fracs:
            conds.append({"id": f"m1:{n}{_fmt_frac(f)}", "arm": "m1", "axis": n, "frac": f})
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


def _gcfg(hcfg, layer):
    return {"steer_layer": layer, "generation": {
        "max_new_tokens": hcfg["generation"]["max_new_tokens"],
        "do_sample": True,
        "temperature": hcfg["generation"]["temperature"],
        "top_p": hcfg["generation"]["top_p"],
    }}


def _intervention(cond, axes, ref, layer, names, cap):
    """(vector, alpha, system) for a condition — the action, never shown to the RM."""
    if cond["arm"] == "m1":
        return torch.tensor(axes[f"{cond['axis']}|{layer}"]), cond["frac"] * ref, None
    if cond["arm"] == "m2":
        return None, 0.0, cond["system"]
    if cond["arm"] == "dense":
        coeffs, _ = project_to_cap(axes, names, layer, cond["coeffs"], cap)
        return combined_vector(axes, names, layer, coeffs), ref, None
    return None, 0.0, None


def _emit(logf, model, tok, rm, rm_tok, mk, cond, seed, batch, comps, compute_nll):
    for p, c in zip(batch, comps):
        logf.write(json.dumps({
            "model": mk, "condition": cond["id"], "arm": cond["arm"], "seed": seed,
            "prompt": p, "completion": c, "rm": rm_score(rm, rm_tok, p, c),
            "nll": mean_completion_nll(model, tok, p, c) if compute_nll else None,
            "words": len(c.split()),
        }) + "\n")
    logf.flush()


def run_model(model, tok, rm, rm_tok, model_key, prompts, conds, hcfg,
              names, axes, ref, layer, seeds, logf, done, compute_nll=True):
    gcfg = _gcfg(hcfg, layer)
    bs = hcfg["generation"]["batch_size"]
    cap = hcfg["grid"]["cap_combined_norm"]
    for cond in conds:
        vec, alpha, system = _intervention(cond, axes, ref, layer, names, cap)
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
                _emit(logf, model, tok, rm, rm_tok, model_key, cond, seed, batch,
                      comps, compute_nll)
            print(f"[{model_key}] {cond['id']} seed {seed}: "
                  f"{len(prompts)} gens ({time.time() - t0:.0f}s)")


def run_validation(model, tok, rm, rm_tok, mk, prompts, conds, hcfg, names,
                   axes, ref, layer, val_seeds, logf, compute_nll, log_path):
    """Selective winner-validation (full mode). Using the seed-0 main grid already
    on disk for this model, pick each prompt's best condition WITHIN each modality
    (none = decline = 0), then regenerate only those winners + none at the held-out
    validation seeds. Debiases the per-prompt oracle at a fraction of a full re-grid."""
    rows = [json.loads(l) for l in open(log_path)
            if f'"model": "{mk}"' in l]
    s0 = {(r["condition"], r["prompt"]): r["rm"] for r in rows if r["seed"] == 0}
    have = collections.Counter((r["condition"], r["seed"]) for r in rows
                               if r["seed"] in val_seeds)
    id2cond = {c["id"]: c for c in conds}
    prefixes = sorted({c["id"].split(":")[0] for c in conds if c["arm"] != "none"})
    # per prompt, per modality: seed-0 winner (skip if 'none' wins — its delta is 0)
    by_cond = {}
    for p in prompts:
        n0 = s0.get(("none", p))
        if n0 is None:
            continue
        for pre in prefixes:
            cids = [c["id"] for c in conds if c["id"].startswith(pre + ":")]
            best, bd = None, 0.0                          # 0.0 = the none option
            for c in cids:
                d = s0.get((c, p))
                if d is not None and d - n0 > bd:
                    bd, best = d - n0, c
            if best is not None:
                by_cond.setdefault(best, []).append(p)
    gcfg = _gcfg(hcfg, layer)
    cap = hcfg["grid"]["cap_combined_norm"]
    bs = hcfg["generation"]["batch_size"]
    jobs = list(by_cond.items()) + [("none", prompts)]    # none for every prompt
    for cid, plist in jobs:
        cond = id2cond[cid] if cid != "none" else {"id": "none", "arm": "none"}
        vec, alpha, system = _intervention(cond, axes, ref, layer, names, cap)
        for seed in val_seeds:
            if have[(cid, seed)] >= len(plist):           # precise resume
                continue
            torch.manual_seed(seed * 1000 + 7)            # distinct CRN stream per val seed
            for i in range(0, len(plist), bs):
                batch = plist[i : i + bs]
                comps = generate_batch(model, tok, batch, gcfg, system=system,
                                       vector=vec, alpha=alpha)
                _emit(logf, model, tok, rm, rm_tok, mk, cond, seed, batch, comps, compute_nll)
        print(f"[{mk}] validate {cid}: {len(plist)} prompts x {len(val_seeds)} seeds")


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


def _metrics(score, none, cond_ids, prompts, seeds):
    """Within-modality paired-vs-none summary. A policy may always decline to
    steer, so the none option (delta 0) is a candidate everywhere:
      static  best SINGLE condition's mean paired delta, or 0 (fixed arm)
      raw     mean over (prompt,seed) of best-of-{conditions, none} (biased max)
      valid   per-prompt winner picked on seeds[0] (incl. none), evaluated on the
              held-out seeds — the debiased conditional headroom
    """
    means = {}
    for c in cond_ids:
        ds = [score(c, s, p) - none(s, p) for p in prompts for s in seeds
              if score(c, s, p) is not None and none(s, p) is not None]
        if ds:
            means[c] = float(np.mean(ds))
    static = max([0.0] + list(means.values()))

    raw = []
    for p in prompts:
        for s in seeds:
            if none(s, p) is None:
                continue
            d = [score(c, s, p) - none(s, p) for c in cond_ids if score(c, s, p) is not None]
            raw.append(max([0.0] + d))

    val, s0 = [], seeds[0]
    for p in prompts:
        if none(s0, p) is None:
            continue
        d0 = {c: score(c, s0, p) - none(s0, p) for c in cond_ids if score(c, s0, p) is not None}
        d0["__none__"] = 0.0
        win = max(d0, key=d0.get)
        if win == "__none__":
            val.append(0.0)
        else:
            ev = [score(win, s, p) - none(s, p) for s in seeds[1:]
                  if score(win, s, p) is not None and none(s, p) is not None]
            if ev:
                val.append(float(np.mean(ev)))
    nan = float("nan")
    return static, (float(np.mean(raw)) if raw else nan), (float(np.mean(val)) if val else nan)


def report_2x2(hcfg, prompts):
    """Conditional steering (M1) vs conditional prompting (M2), per model, both RMs."""
    r1 = [json.loads(l) for l in open(OUT / "headroom_log.jsonl")]
    r2 = {(x["model"], x["condition"], x["seed"], x["prompt"]): x["rm2"]
          for x in (json.loads(l) for l in open(OUT / "headroom_rm2.jsonl"))}
    seeds = hcfg["pilot"]["seeds"]
    lines = ["# A2 2x2 prototype — conditional steering vs prompting\n",
             f"{len(prompts)} prompts, seeds {seeds}. Within-modality paired ΔRM vs "
             "`none`; a conditional policy may always decline to steer (none = 0). "
             "static = best single fixed condition; valid-oracle = per-prompt best "
             f"picked on seed {seeds[0]}, evaluated on {seeds[1:]} (debiased). "
             "RM1 = Skywork-Qwen3-0.6B, RM2 = Skywork-Llama-1B.\n"]
    for mk in sorted({x["model"] for x in r1}):
        mr = [x for x in r1 if x["model"] == mk]
        rm1 = {(x["condition"], x["seed"], x["prompt"]): x["rm"] for x in mr}
        none1 = lambda s, p: rm1.get(("none", s, p))
        none2 = lambda s, p, mk=mk: r2.get((mk, "none", s, p))
        sc1 = lambda c, s, p: rm1.get((c, s, p))
        sc2 = lambda c, s, p, mk=mk: r2.get((mk, c, s, p))
        conds = [c for c in dict.fromkeys(x["condition"] for x in mr) if c != "none"]
        mods = [("M1 steering", [c for c in conds if c.startswith("m1:")]),
                ("M2 prompting", [c for c in conds if c.startswith("m2:")]),
                ("dense", [c for c in conds if c.startswith("dense:")])]
        lines.append(f"\n## {mk}\n")
        lines.append("| modality | n | static (fixed) | raw-oracle | **valid-oracle RM1** | valid-oracle RM2 |")
        lines.append("|---|---|---|---|---|---|")
        for mod, cids in mods:
            if not cids:
                continue
            sb, ro, vo1 = _metrics(sc1, none1, cids, prompts, seeds)
            _, _, vo2 = _metrics(sc2, none2, cids, prompts, seeds)
            lines.append(f"| {mod} | {len(cids)} | {sb:+.2f} | {ro:+.2f} | **{vo1:+.2f}** | {vo2:+.2f} |")
    lines.append(
        "\n## Reading\nvalid-oracle is the debiased per-prompt headroom a CONDITIONAL "
        "policy could capture (0 = conditioning captures nothing beyond declining to "
        "steer). Compare within a model: does M1 (steering) reach what M2 (prompting) "
        "does? valid-oracle > static means conditioning beats any fixed choice. Agreement "
        "between RM1 and RM2 valid-oracle guards against tuning to one RM's noise."
    )
    (BASIS / "headroom_2x2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport -> {BASIS / 'headroom_2x2.md'}")


def _tag(p):
    """Cheap prompt tags for the 'where does headroom live' breakdown."""
    low = p.lower()
    task = any(w in low for w in ("write", "code", "function", "translate",
                                  "calculate", "solve", "convert", "list ", "sql"))
    return {"len": "short" if len(p.split()) < 30 else "long",
            "form": "question" if "?" in p else "statement",
            "kind": "task" if task else "open"}


def _boot(vals, n=2000, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = np.array(vals, float)
    ms = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(a.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def _valid_oracle(cids, prompts, sel, ev):
    """Winner picked by `sel(c,p)` seed-0 delta (none-decline = 0), evaluated by
    `ev(c,p)` over the validation seeds. Returns {prompt: value}."""
    out = {}
    for p in prompts:
        best, bd = None, 0.0
        for c in cids:
            d = sel(c, p)
            if d is not None and d > bd:
                bd, best = d, c
        if best is None:
            out[p] = 0.0                                  # declined to steer
        else:
            e = ev(best, p)
            if e is not None:
                out[p] = e
    return out


def report_full(hcfg, prompts):
    """Full-run report: static + debiased conditional headroom per model x modality,
    both RMs, with bootstrap CIs and a where-it-lives tag breakdown."""
    val_seeds = hcfg["full"]["validation_seeds"]
    r1 = [json.loads(l) for l in open(FULL_LOG)]
    r2 = {(x["model"], x["condition"], x["seed"], x["prompt"]): x["rm2"]
          for x in (json.loads(l) for l in open(FULL_RM2))} \
        if FULL_RM2.exists() else {}
    lines = ["# A2 full run — steering vs prompting, 200 prompts\n",
             f"{len(prompts)} prompts, main seed 0 + winner-validation on {val_seeds}. "
             "Within-modality paired ΔRM vs none; conditional policy may decline "
             "(none = 0). static = best fixed condition (seed 0); valid-oracle = "
             "per-prompt winner picked on seed 0 by RM1, evaluated out-of-seed. "
             "[95% bootstrap CI over prompts]. RM1 = Qwen3-0.6B, RM2 = Llama-1B.\n"]
    hero = {}
    for mk in sorted({x["model"] for x in r1}):
        mr = [x for x in r1 if x["model"] == mk]
        rm1 = {(x["condition"], x["seed"], x["prompt"]): x["rm"] for x in mr}
        n1 = lambda s, p: rm1.get(("none", s, p))
        n2 = lambda s, p, mk=mk: r2.get((mk, "none", s, p))
        d1s0 = lambda c, p: (None if rm1.get((c, 0, p)) is None or n1(0, p) is None
                             else rm1[(c, 0, p)] - n1(0, p))
        def d1val(c, p):
            e = [rm1[(c, s, p)] - n1(s, p) for s in val_seeds
                 if rm1.get((c, s, p)) is not None and n1(s, p) is not None]
            return np.mean(e) if e else None
        def d2val(c, p, mk=mk):
            e = [r2[(mk, c, s, p)] - n2(s, p) for s in val_seeds
                 if r2.get((mk, c, s, p)) is not None and n2(s, p) is not None]
            return np.mean(e) if e else None
        conds0 = [c for c in {x["condition"] for x in mr if x["seed"] == 0} if c != "none"]
        lines.append(f"\n## {mk}\n")
        lines.append("| modality | n | static (RM1) | **valid-oracle RM1 [CI]** | valid-oracle RM2 [CI] |")
        lines.append("|---|---|---|---|---|")
        for label, pre in (("M1 steering", "m1"), ("M2 prompting", "m2"), ("dense", "dense")):
            cids = [c for c in conds0 if c.startswith(pre + ":")]
            if not cids:
                continue
            means = {c: np.mean([d1s0(c, p) for p in prompts if d1s0(c, p) is not None])
                     for c in cids}
            static = max([0.0] + list(means.values()))
            vo1 = _valid_oracle(cids, prompts, d1s0, d1val)
            vo2 = _valid_oracle(cids, prompts, d1s0, d2val)
            m1, lo1, hi1 = _boot(list(vo1.values()))
            m2, lo2, hi2 = _boot(list(vo2.values()))
            hero[(mk, pre)] = vo1
            lines.append(f"| {label} | {len(cids)} | {static:+.2f} | "
                         f"**{m1:+.2f} [{lo1:+.2f},{hi1:+.2f}]** | {m2:+.2f} [{lo2:+.2f},{hi2:+.2f}] |")

    # where headroom lives: valid-oracle RM1 by prompt tag, for the 7B arms
    lines.append("\n## Where headroom lives (7B valid-oracle RM1 by prompt tag)\n")
    lines.append("| tag | M1 steering | M2 prompting | n |")
    lines.append("|---|---|---|---|")
    tagvals = {}
    for dim in ("len", "form", "kind"):
        for val in sorted({_tag(p)[dim] for p in prompts}):
            sub = [p for p in prompts if _tag(p)[dim] == val]
            row = {}
            for pre in ("m1", "m2"):
                if ("large", pre) in hero:
                    vo = hero[("large", pre)]
                    vals = [vo[p] for p in sub if p in vo]
                    row[pre] = np.mean(vals) if vals else float("nan")
            lines.append(f"| {val} | {row.get('m1', float('nan')):+.2f} | "
                         f"{row.get('m2', float('nan')):+.2f} | {len(sub)} |")
    lines.append(
        "\n## Reading\nvalid-oracle = debiased ceiling a conditional policy could "
        "capture (a learned policy reaches ~55–90% of it, per B0). CI excluding 0 = "
        "real headroom. RM1 vs RM2 agreement = not one-RM noise. Compare M1 (steering) "
        "vs M2 (prompting) within the 7B; compare 1.7B-M2 vs 7B for the cost story."
    )
    (BASIS / "headroom_full.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport -> {BASIS / 'headroom_full.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--model", default="both", choices=["base", "large", "both"])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report-2x2", action="store_true")
    ap.add_argument("--report-full", action="store_true")
    ap.add_argument("--nll", action="store_true", help="force the perplexity pass on")
    args = ap.parse_args()

    cfg = load_config()
    hcfg = load_hr_config()
    device = resolve_device(cfg)
    names = [a["name"] for a in load_basis_config()["axes"]]
    modecfg = hcfg[args.mode]
    data = json.load(open(REPO_ROOT / "data" / "prompts.json"))
    prompts = data["train"][: modecfg["n_prompts"]]
    seeds = modecfg["seeds"]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        report_pilot(hcfg, prompts)
        return
    if args.report_2x2:
        report_2x2(hcfg, prompts)
        return
    if args.report_full:
        report_full(hcfg, prompts)
        return

    # perplexity pass: off in full mode (fluency is not a gate) unless forced
    compute_nll = args.nll or args.mode == "pilot"
    arms_override = modecfg.get("model_arms", {})
    rm, rm_tok = load_rm(cfg, device)
    log_path = FULL_LOG if args.mode == "full" else OUT / "headroom_log.jsonl"
    model_keys = ["base", "large"] if args.model == "both" else [args.model]

    t0 = time.time()
    with open(log_path, "a") as logf:
        for mk in model_keys:
            mcfg = hcfg["models"][mk]
            steer = mcfg["steer"]
            m_layer, m_axes = steer["layer"], np.load(BASIS / steer["axes"])
            arms = arms_override.get(mk, mcfg["arms"])
            conds = build_conditions(hcfg, names, arms, steer["alpha_fracs"])
            done = _complete_keys(log_path, mk, len(prompts))
            print(f"=== {mk} [{args.mode}]: {len(conds)} conds x {len(seeds)} seed(s) "
                  f"(layer {m_layer}, {steer['axes']}, arms {arms}, {len(done)} done) ===")
            if mk == "base":
                model, tok = load_base(cfg, device)
            else:
                model, tok = load_named(mcfg["model"], cfg, device)
            ref = (measure_ref_norm(model, tok, prompts[:8], m_layer)
                   if {"m1", "dense"} & set(arms) else None)
            print(f"ref norm {ref:.1f}" if ref else "no steering arms")
            run_model(model, tok, rm, rm_tok, mk, prompts, conds, hcfg,
                      names, m_axes, ref, m_layer, seeds, logf, done, compute_nll)
            if args.mode == "full":
                run_validation(model, tok, rm, rm_tok, mk, prompts, conds, hcfg,
                               names, m_axes, ref, m_layer,
                               modecfg["validation_seeds"], logf, compute_nll, log_path)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(log_cost("A2", f"headroom_{args.mode}_{args.model}", time.time() - t0, device))
    if args.mode == "pilot":
        report_pilot(hcfg, prompts)


if __name__ == "__main__":
    main()
