"""Stage A step 3: sanity-steering sweep on held-out prompts.

For every axis and steering magnitude (± fractions of the measured residual
norm), generate steered completions, then score ALL seven proxies plus the RM
on each one. Products:
  results/sanity/baseline.jsonl + {axis}.jsonl   raw sweep data
  basis/sanity_samples.md                        qualitative sheets (READ FIRST)
  basis/sanity_report.md                         own-proxy monotonicity tables,
                                                 RM-vs-alpha, cross-steering matrix

The cross-steering matrix is the second half of the pre-registered collapse
rule (PLAN.md): steering axis i should move proxy i much more than proxy j.
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
    load_rm,
    log_cost,
    resolve_device,
    rm_score,
)
from proxies import PROXIES

OUT = REPO_ROOT / "results" / "sanity"
BASIS = REPO_ROOT / "basis"


def measure_ref_norm(model, tok, prompts, layer):
    """Mean per-token L2 norm of the residual stream at `layer` (prompt tokens)."""
    captured = {}

    def hook(_m, _i, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    norms = []
    try:
        with torch.no_grad():
            for p in prompts:
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)
                model(enc["input_ids"])
                norms.append(captured["h"][0].float().norm(dim=-1).mean().item())
    finally:
        handle.remove()
    return float(np.mean(norms))


def score_rows(rm, rm_tok, prompts, completions, axis, frac):
    rows = []
    for prompt, completion in zip(prompts, completions):
        rows.append(
            {
                "axis": axis,
                "alpha_frac": frac,
                "prompt": prompt,
                "completion": completion,
                "proxies": {name: fn(completion) for name, fn in PROXIES.items()},
                "rm": rm_score(rm, rm_tok, prompt, completion),
            }
        )
    return rows


def run_condition(model, tok, rm, rm_tok, prompts, cfg, scfg, axis, vector, frac, ref):
    rows = []
    bs = scfg["batch_size"]
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        completions = generate_batch(
            model, tok, batch, cfg, vector=vector, alpha=frac * ref
        )
        rows.extend(score_rows(rm, rm_tok, batch, completions, axis, frac))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", nargs="*", default=[],
                        help="axis names whose sweep files should be regenerated")
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config()
    bcfg = load_basis_config()
    scfg = bcfg["sanity"]
    device = resolve_device(cfg)
    torch.manual_seed(bcfg["data"]["seed"])
    layer = cfg["steer_layer"]

    model, tok = load_base(cfg, device)
    rm, rm_tok = load_rm(cfg, device)
    prompts = json.load(open(REPO_ROOT / "data" / "prompts.json"))["heldout"]
    axes = np.load(BASIS / "axes.npz")
    axis_names = [a["name"] for a in bcfg["axes"]]

    ref = measure_ref_norm(model, tok, prompts[: scfg["ref_norm_prompts"]], layer)
    print(f"reference residual norm at layer {layer}: {ref:.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT / "baseline.jsonl"
    if not baseline_path.exists():
        rows = run_condition(
            model, tok, rm, rm_tok, prompts, cfg, scfg, "baseline", None, 0.0, ref
        )
        with open(baseline_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("baseline done")

    for name in axis_names:
        path = OUT / f"{name}.jsonl"
        if path.exists() and name in args.redo:
            path.rename(path.with_suffix(".old.jsonl.bak"))
            print(f"{name}: --redo, regenerating (old file -> .old.jsonl.bak)")
        elif path.exists():
            print(f"{name}: exists, skipping")
            continue
        vec = torch.tensor(axes[f"{name}|{layer}"])
        rows = []
        for frac in scfg["alpha_fracs"]:
            rows.extend(
                run_condition(
                    model, tok, rm, rm_tok, prompts, cfg, scfg, name, vec, frac, ref
                )
            )
            print(f"  {name} frac={frac} done")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write_reports(bcfg, scfg, axis_names, layer, ref)
    print(log_cost("A", "sanity_sweep", time.time() - t0, device,
                   notes=f"layer={layer} ref_norm={ref:.1f}"))


def write_reports(bcfg, scfg, axis_names, layer, ref):
    baseline = [json.loads(l) for l in open(OUT / "baseline.jsonl")]
    data = {n: [json.loads(l) for l in open(OUT / f"{n}.jsonl")] for n in axis_names}
    fracs = [0.0] + scfg["alpha_fracs"]
    fracs_sorted = sorted(fracs)

    def condition(name, frac):
        return baseline if frac == 0.0 else [r for r in data[name] if r["alpha_frac"] == frac]

    report = [f"# Sanity steering report (layer {layer}, ref norm {ref:.1f})\n"]
    report.append("Own-proxy mean and RM mean vs alpha fraction, per axis:\n")
    for name in axis_names:
        report.append(f"\n## {name}\n")
        report.append("| alpha_frac | own proxy | RM | mean words |")
        report.append("|---|---|---|---|")
        for frac in fracs_sorted:
            rows = condition(name, frac)
            own = np.mean([r["proxies"][name] for r in rows])
            rmv = np.mean([r["rm"] for r in rows])
            words = np.mean([len(r["completion"].split()) for r in rows])
            report.append(f"| {frac:+.2f} | {own:.2f} | {rmv:.2f} | {words:.0f} |")

    hi = scfg.get("cross_frac", 0.2)
    report.append(
        f"\n## Cross-steering matrix (z-scored effect at alpha_frac ±{hi})\n"
    )
    report.append(
        "Entry (i,j): z-scored change in proxy j between steering axis i at "
        f"+{hi} vs -{hi}. Diagonal should dominate its row.\n"
    )
    base_std = {
        n: np.std([r["proxies"][n] for r in baseline]) + 1e-6 for n in axis_names
    }
    report.append("| steer \\ proxy | " + " | ".join(n[:12] for n in axis_names) + " |")
    report.append("|---|" + "---|" * len(axis_names))
    for name in axis_names:
        pos = [r["proxies"] for r in condition(name, hi)]
        neg = [r["proxies"] for r in condition(name, -hi)]
        cells = []
        for other in axis_names:
            dz = (np.mean([p[other] for p in pos]) - np.mean([p[other] for p in neg])) / base_std[other]
            cells.append(f"{dz:+.1f}")
        report.append(f"| {name[:12]} | " + " | ".join(cells) + " |")

    with open(BASIS / "sanity_report.md", "w") as f:
        f.write("\n".join(report) + "\n")

    # Qualitative sheets: same few prompts across the whole sweep, per axis.
    sheet = ["# Sanity steering samples (read before the report tables)\n"]
    show = baseline[: scfg["samples_per_axis"]]
    for name in axis_names:
        sheet.append(f"\n## {name}\n")
        for b in show:
            sheet.append(f"**prompt:** {b['prompt'][:150]}\n")
            for frac in fracs_sorted:
                rows = [r for r in condition(name, frac) if r["prompt"] == b["prompt"]]
                if rows:
                    text = rows[0]["completion"][:300].replace("\n", " ")
                    sheet.append(f"- **{frac:+.2f}** {text}")
            sheet.append("")
    with open(BASIS / "sanity_samples.md", "w") as f:
        f.write("\n".join(sheet) + "\n")
    print(f"reports -> {BASIS / 'sanity_report.md'}, {BASIS / 'sanity_samples.md'}")


if __name__ == "__main__":
    main()
