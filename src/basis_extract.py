"""Stage A step 2: contrastive pair generation + basis extraction.

Phases (run in order; each is resumable and logs measured cost):
  prompts     sample + cache train/held-out UltraFeedback prompts
  compliance  20 pairs/axis: can the base model act out the poles when told to?
              (early escalation trigger — inspect before running 'generate')
  generate    full 200 prompts x 7 axes x 2 poles pair generation
  extract     teacher-forced activation capture -> axis vectors + cosine report
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset

from models import (
    REPO_ROOT,
    generate_batch,
    load_base,
    load_config,
    log_cost,
    mean_completion_activations,
    resolve_device,
)
from proxies import PROXIES

DATA = REPO_ROOT / "data"
BASIS = REPO_ROOT / "basis"


def load_basis_config():
    with open(REPO_ROOT / "configs" / "basis.yaml") as f:
        return yaml.safe_load(f)


def phase_prompts(cfg, bcfg):
    out = DATA / "prompts.json"
    if out.exists():
        print(f"{out} exists, skipping")
        return
    d, bd = cfg["data"], bcfg["data"]
    n_total = bd["n_train"] + bd["n_heldout"]
    stream = load_dataset(d["prompt_dataset"], split=d["prompt_split"], streaming=True)
    stream = stream.shuffle(seed=bd["seed"], buffer_size=2000)
    prompts, seen = [], set()
    for row in stream:
        p = row[d["prompt_field"]].strip()
        if 10 <= len(p) <= bd["max_prompt_chars"] and p not in seen:
            seen.add(p)
            prompts.append(p)
        if len(prompts) == n_total:
            break
    random.Random(bd["seed"]).shuffle(prompts)
    DATA.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"train": prompts[: bd["n_train"]], "heldout": prompts[bd["n_train"]:]},
            f,
            indent=1,
        )
    print(f"cached {bd['n_train']} train + {bd['n_heldout']} heldout -> {out}")


def _gen_pairs(model, tok, cfg, bcfg, axis, prompts, out_path):
    """Generate pole completions for one axis; resumable via line count."""
    done = 0
    if out_path.exists():
        done = sum(1 for _ in open(out_path))
    todo = prompts[done // 2:]  # 2 lines (pos+neg) per prompt
    bs = bcfg["pairs"]["batch_size"]
    gen_cfg = {**cfg, "generation": {**cfg["generation"], "max_new_tokens": bcfg["pairs"]["max_new_tokens"]}}
    proxy = PROXIES[axis["name"]]
    with open(out_path, "a") as f:
        for i in range(0, len(todo), bs):
            batch = todo[i : i + bs]
            for pole, instruction in (
                ("pos", axis["pos_instruction"]),
                ("neg", axis["neg_instruction"]),
            ):
                completions = generate_batch(
                    model, tok, batch, gen_cfg, system=instruction
                )
                for prompt, completion in zip(batch, completions):
                    f.write(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "pole": pole,
                                "completion": completion,
                                "proxy": proxy(completion),
                            }
                        )
                        + "\n"
                    )
            f.flush()
            print(f"  {axis['name']}: {done // 2 + i + len(batch)}/{len(prompts)} prompts")


def _proxy_summary(path):
    rows = [json.loads(line) for line in open(path)]
    pos = [r["proxy"] for r in rows if r["pole"] == "pos"]
    neg = [r["proxy"] for r in rows if r["pole"] == "neg"]
    return np.mean(pos), np.mean(neg), rows


def phase_compliance(cfg, bcfg, model, tok, axes=None):
    if axes:
        bcfg = {**bcfg, "axes": [a for a in bcfg["axes"] if a["name"] in axes]}
    prompts = json.load(open(DATA / "prompts.json"))["train"][
        : bcfg["pairs"]["compliance_n"]
    ]
    comp_dir = DATA / "compliance"
    comp_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Pole-compliance check (instructed generations, no steering)\n"]
    table = ["| axis | mean proxy (pos) | mean proxy (neg) | separation ok? |", "|---|---|---|---|"]
    for axis in bcfg["axes"]:
        path = comp_dir / f"{axis['name']}.jsonl"
        _gen_pairs(model, tok, cfg, bcfg, axis, prompts, path)
        mp, mn, rows = _proxy_summary(path)
        table.append(f"| {axis['name']} | {mp:.2f} | {mn:.2f} | {'YES' if mp > mn else 'NO'} |")
        lines.append(f"\n## {axis['name']}  (pos={axis['pos']}, neg={axis['neg']})\n")
        for r in [r for r in rows if r["pole"] == "pos"][:3]:
            lines.append(f"**[{axis['pos']}]** {r['prompt'][:120]}\n> {r['completion'][:500]}\n")
        for r in [r for r in rows if r["pole"] == "neg"][:3]:
            lines.append(f"**[{axis['neg']}]** {r['prompt'][:120]}\n> {r['completion'][:500]}\n")
    sheet_name = "samples.md" if not axes else f"samples_{'_'.join(sorted(axes))}.md"
    with open(comp_dir / sheet_name, "w") as f:
        f.write("\n".join(table) + "\n\n" + "\n".join(lines))
    print("\n".join(table))
    print(f"\nsample sheet -> {comp_dir / sheet_name}")


def phase_generate(cfg, bcfg, model, tok):
    prompts = json.load(open(DATA / "prompts.json"))["train"]
    pairs_dir = DATA / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for axis in bcfg["axes"]:
        print(f"axis {axis['name']}")
        _gen_pairs(model, tok, cfg, bcfg, axis, prompts, pairs_dir / f"{axis['name']}.jsonl")


def phase_extract(cfg, bcfg, model, tok):
    layers = bcfg["capture_layers"]
    BASIS.mkdir(exist_ok=True)
    vectors = {}       # (axis, layer) -> unit vector
    per_axis_stats = []
    for axis in bcfg["axes"]:
        name = axis["name"]
        rows = [json.loads(line) for line in open(DATA / "pairs" / f"{name}.jsonl")]
        sums = {(pole, layer): None for pole in ("pos", "neg") for layer in layers}
        counts = {"pos": 0, "neg": 0}
        for r in rows:
            if not r["completion"].strip():
                continue
            acts = mean_completion_activations(
                model, tok, r["prompt"], r["completion"], layers
            )
            counts[r["pole"]] += 1
            for layer in layers:
                key = (r["pole"], layer)
                sums[key] = acts[layer] if sums[key] is None else sums[key] + acts[layer]
        for layer in layers:
            diff = sums[("pos", layer)] / counts["pos"] - sums[("neg", layer)] / counts["neg"]
            vectors[(name, layer)] = (diff / diff.norm()).numpy()
        mp, mn, _ = _proxy_summary(DATA / "pairs" / f"{name}.jsonl")
        per_axis_stats.append((name, counts["pos"], counts["neg"], mp, mn))
        print(f"{name}: n={counts} proxy pos={mp:.2f} neg={mn:.2f}")

    np.savez(
        BASIS / "axes.npz",
        **{f"{name}|{layer}": v for (name, layer), v in vectors.items()},
    )
    axis_names = [a["name"] for a in bcfg["axes"]]
    report = ["# Basis extraction report\n"]
    report.append("| axis | n_pos | n_neg | proxy pos | proxy neg |")
    report.append("|---|---|---|---|---|")
    for name, npos, nneg, mp, mn in per_axis_stats:
        report.append(f"| {name} | {npos} | {nneg} | {mp:.2f} | {mn:.2f} |")
    for layer in layers:
        mat = np.stack([vectors[(n, layer)] for n in axis_names])
        cos = mat @ mat.T
        report.append(f"\n## Cosine similarity, layer {layer}\n")
        report.append("| | " + " | ".join(n[:12] for n in axis_names) + " |")
        report.append("|---|" + "---|" * len(axis_names))
        for i, n in enumerate(axis_names):
            report.append(
                f"| {n[:12]} | "
                + " | ".join(f"{cos[i, j]:.2f}" for j in range(len(axis_names)))
                + " |"
            )
        flagged = [
            (axis_names[i], axis_names[j], cos[i, j])
            for i in range(len(axis_names))
            for j in range(i + 1, len(axis_names))
            if abs(cos[i, j]) >= 0.7
        ]
        if flagged and layer == 21:
            report.append("\n**COLLAPSE-RULE FLAGS (|cos| >= 0.7 at default layer):** " + str(flagged))
    with open(BASIS / "extraction_report.md", "w") as f:
        f.write("\n".join(report) + "\n")
    with open(BASIS / "metadata.yaml", "w") as f:
        yaml.safe_dump(
            {
                "date": time.strftime("%Y-%m-%d"),
                "base_model": cfg["base_model"],
                "capture_layers": layers,
                "default_layer": cfg["steer_layer"],
                "n_train_prompts": bcfg["data"]["n_train"],
                "capture": "teacher-forced, instruction-free context, mean over completion tokens",
                "axes": axis_names,
            },
            f,
        )
    print(f"basis -> {BASIS / 'axes.npz'}; report -> {BASIS / 'extraction_report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["prompts", "compliance", "generate", "extract"])
    parser.add_argument("--axes", nargs="*", default=None,
                        help="compliance only: restrict to these axis names")
    args = parser.parse_args()
    t0 = time.time()
    cfg = load_config()
    bcfg = load_basis_config()
    device = resolve_device(cfg)
    torch.manual_seed(bcfg["data"]["seed"])

    if args.phase == "prompts":
        phase_prompts(cfg, bcfg)
    else:
        model, tok = load_base(cfg, device)
        if args.phase == "compliance":
            phase_compliance(cfg, bcfg, model, tok, axes=args.axes)
        else:
            {"generate": phase_generate, "extract": phase_extract}[args.phase](
                cfg, bcfg, model, tok
            )
    print(log_cost("A", f"basis_{args.phase}", time.time() - t0, device))


if __name__ == "__main__":
    main()
