"""Stage B: fixed global steering arm.

Search: CEM over a in R^k (alpha-fraction per axis, combined-vector norm
capped) maximizing mean RM reward on a fixed train subset. Round 1 always
includes the zero action (SFT) and the Stage-A-informed prior mean, so
"steering doesn't help" is observable, not assumed.

Eval: best action vs SFT on the 50 held-out prompts across sampling seeds,
with drift/diversity guards: distinct-n, mean length, per-token
KL(steered || base) computed teacher-forced on the steered completions.

Outputs: results/fixed_steer/{search_log.jsonl, cem_state.json, eval.jsonl},
basis/stageB_report.md, basis/stageB_samples.md.
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from basis_extract import load_basis_config
from models import (
    REPO_ROOT,
    generate_batch,
    load_base,
    load_config,
    load_rm,
    log_cost,
    resolve_device,
    rm_score,
    steering_hook,
)
from steer_sanity import measure_ref_norm

OUT = REPO_ROOT / "results" / "fixed_steer"
BASIS = REPO_ROOT / "basis"


def load_fs_config():
    with open(REPO_ROOT / "configs" / "fixed_steer.yaml") as f:
        return yaml.safe_load(f)


def combined_vector(axes, names, layer, coeffs):
    return sum(
        float(c) * torch.tensor(axes[f"{n}|{layer}"]) for n, c in zip(names, coeffs)
    )


def project_to_cap(axes, names, layer, coeffs, cap):
    """Scale coefficients so the combined added vector's norm <= cap."""
    norm = combined_vector(axes, names, layer, coeffs).norm().item()
    if norm > cap:
        coeffs = coeffs * (cap / norm)
        norm = cap
    return coeffs, norm


def score_action(model, tok, rm, rm_tok, prompts, cfg, bs, vec, ref):
    scores = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        comps = generate_batch(model, tok, batch, cfg, vector=vec, alpha=ref)
        scores.extend(rm_score(rm, rm_tok, p, c) for p, c in zip(batch, comps))
    return float(np.mean(scores))


def search(model, tok, rm, rm_tok, cfg, names, axes, layer, ref, prompts, scfg):
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "search_log.jsonl"
    done = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    k = len(names)
    mean = np.array([scfg["init_mean"].get(n, 0.0) for n in names])
    std = np.full(k, scfg["init_std"])
    subset = prompts[: scfg["train_subset"]]
    cap = scfg["cap_combined_norm"]

    for r, n_cand in enumerate(scfg["rounds"]):
        rng = np.random.default_rng(scfg["seed"] * 100 + r)
        cands = mean + std * rng.standard_normal((n_cand, k))
        if r == 0:
            cands[0] = np.zeros(k)                              # SFT reference
            cands[1] = np.array([scfg["init_mean"].get(n, 0.0) for n in names])
        this_round = [d for d in done if d["round"] == r]
        with open(log_path, "a") as f:
            for idx, c in enumerate(cands):
                if any(d["idx"] == idx for d in this_round):
                    continue
                c, norm = project_to_cap(axes, names, layer, c, cap)
                vec = combined_vector(axes, names, layer, c)
                score = score_action(
                    model, tok, rm, rm_tok, subset, cfg, 8, vec, ref
                )
                row = {
                    "round": r,
                    "idx": idx,
                    "action": dict(zip(names, [round(float(x), 4) for x in c])),
                    "combined_norm": round(norm, 4),
                    "mean_rm": round(score, 4),
                }
                this_round.append(row)
                done.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"round {r} cand {idx}: rm={score:.3f} norm={norm:.3f}")
        n_elite = max(2, int(len(this_round) * scfg["elite_frac"]))
        elites = sorted(this_round, key=lambda d: -d["mean_rm"])[:n_elite]
        emat = np.array([[e["action"][n] for n in names] for e in elites])
        mean, std = emat.mean(axis=0), emat.std(axis=0) + 0.01
        print(f"round {r} elite mean rm={np.mean([e['mean_rm'] for e in elites]):.3f}")

    best = max(done, key=lambda d: d["mean_rm"])
    with open(OUT / "cem_state.json", "w") as f:
        json.dump({"best": best, "final_mean": list(mean), "names": names}, f, indent=1)
    return best


@torch.no_grad()
def mean_token_kl(model, tok, prompt, completion, vec, alpha, layer):
    """Mean per-token KL(steered || base) over completion positions."""
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    comp = tok(completion, return_tensors="pt", add_special_tokens=False)
    ids = torch.cat([prefix["input_ids"], comp["input_ids"]], dim=1).to(model.device)
    n_prefix = prefix["input_ids"].shape[1]
    logits_base = model(ids).logits[0].float()
    with steering_hook(model, layer, vec, alpha):
        logits_steer = model(ids).logits[0].float()
    sl = logits_steer[n_prefix - 1 : -1]
    bl = logits_base[n_prefix - 1 : -1]
    kl = F.kl_div(
        F.log_softmax(bl, dim=-1), F.log_softmax(sl, dim=-1),
        log_target=True, reduction="none",
    ).sum(-1)
    return kl.mean().item()


def distinct_n(completions, n):
    grams, total = set(), 0
    for c in completions:
        toks = c.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return len(grams) / max(total, 1)


def evaluate(model, tok, rm, rm_tok, cfg, names, axes, layer, ref, heldout, fscfg, best):
    ecfg = fscfg["eval"]
    coeffs = np.array([best["action"][n] for n in names])
    vec = combined_vector(axes, names, layer, coeffs)
    rows = []
    for seed in ecfg["gen_seeds"]:
        for arm, v in (("sft", None), ("fixed", vec)):
            torch.manual_seed(seed)
            for i in range(0, len(heldout), 8):
                batch = heldout[i : i + 8]
                comps = generate_batch(
                    model, tok, batch, cfg, vector=v, alpha=ref if v is not None else 0.0
                )
                for p, c in zip(batch, comps):
                    rows.append(
                        {"seed": seed, "arm": arm, "prompt": p, "completion": c,
                         "rm": rm_score(rm, rm_tok, p, c)}
                    )
        print(f"seed {seed} eval done")
    with open(OUT / "eval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    kls = [
        mean_token_kl(model, tok, r["prompt"], r["completion"], vec, ref, layer)
        for r in rows
        if r["arm"] == "fixed" and r["seed"] == ecfg["gen_seeds"][0]
    ]

    report = ["# Stage B report — fixed global steering\n"]
    report.append(f"Best action (combined norm {best['combined_norm']}):\n")
    for n in names:
        report.append(f"- {n}: {best['action'][n]:+.3f}")
    report.append("\n| arm | seed | mean RM | mean words | distinct-1 | distinct-2 |")
    report.append("|---|---|---|---|---|---|")
    summary = {}
    for arm in ("sft", "fixed"):
        for seed in ecfg["gen_seeds"]:
            sel = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            comps = [r["completion"] for r in sel]
            mrm = np.mean([r["rm"] for r in sel])
            summary.setdefault(arm, []).append(mrm)
            report.append(
                f"| {arm} | {seed} | {mrm:.3f} | "
                f"{np.mean([len(c.split()) for c in comps]):.0f} | "
                f"{distinct_n(comps, 1):.3f} | {distinct_n(comps, 2):.3f} |"
            )
    report.append(
        f"\nmean KL(steered||base) per token, fixed arm seed {ecfg['gen_seeds'][0]}: "
        f"{np.mean(kls):.4f} nats"
    )
    d = np.mean(summary["fixed"]) - np.mean(summary["sft"])
    report.append(f"\n**fixed − sft held-out mean RM: {d:+.3f}**")
    with open(BASIS / "stageB_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    sheet = ["# Stage B samples — sft vs fixed (same prompt, same seed)\n"]
    seed0 = ecfg["gen_seeds"][0]
    for p in heldout[: ecfg["samples_sheet_n"]]:
        sheet.append(f"**prompt:** {p[:150]}\n")
        for arm in ("sft", "fixed"):
            r = next(
                x for x in rows
                if x["arm"] == arm and x["seed"] == seed0 and x["prompt"] == p
            )
            sheet.append(f"- **{arm}** (rm {r['rm']:.2f}): {r['completion'][:350].replace(chr(10), ' ')}")
        sheet.append("")
    with open(BASIS / "stageB_samples.md", "w") as f:
        f.write("\n".join(sheet) + "\n")
    print(f"reports -> {BASIS / 'stageB_report.md'}, {BASIS / 'stageB_samples.md'}")


def main():
    t0 = time.time()
    cfg = load_config()
    bcfg = load_basis_config()
    fscfg = load_fs_config()
    scfg = fscfg["search"]
    device = resolve_device(cfg)
    layer = cfg["steer_layer"]
    names = [a["name"] for a in bcfg["axes"]]

    model, tok = load_base(cfg, device)
    rm, rm_tok = load_rm(cfg, device)
    axes = np.load(BASIS / "axes.npz")
    data = json.load(open(REPO_ROOT / "data" / "prompts.json"))
    ref = measure_ref_norm(model, tok, data["train"][:8], layer)
    print(f"ref norm {ref:.1f}, axes: {names}")

    state_path = OUT / "cem_state.json"
    if state_path.exists():
        best = json.load(open(state_path))["best"]
        print(f"search already complete, best rm={best['mean_rm']}")
    else:
        best = search(model, tok, rm, rm_tok, cfg, names, axes, layer, ref,
                      data["train"], scfg)
    print(f"best action: {best['action']} (rm {best['mean_rm']})")
    evaluate(model, tok, rm, rm_tok, cfg, names, axes, layer, ref,
             data["heldout"], fscfg, best)
    print(log_cost("B", "fixed_steer", time.time() - t0, device,
                   notes=f"best_rm={best['mean_rm']}"))


if __name__ == "__main__":
    main()
