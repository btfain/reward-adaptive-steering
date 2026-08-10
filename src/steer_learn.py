"""Subproject 1 / S1.1 — RWR reward-driven steering-vector learner (synthetic positive control).

Learns a steering direction (residual-stream injection at layer L) by reward-weighted
regression toward the KL-optimal tilt: per prompt, draw a base-completion pool, fix tilt
weights w_k = softmax(R_k / beta), and minimize the weighted teacher-forced NLL of the
STEERED policy. Fully differentiable in the injection (no sampling in the inner loop);
the pool + rewards are precomputed. See specs/subproject1_spec.md.

S1.1 uses an ANALYTIC reward on cheap phi features with a KNOWN target direction e*, so
we can check the learner recovers a reward-aligned steering vector before any real RM
(machinery check, a la B0). rank>1 and the mlp controller are stubbed for later gates.

Phases: pool (GPU generate + phi) -> learn (gradient) -> eval (on-policy delta-R + phi recovery).

    python src/steer_learn.py --phase all
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from models import (
    REPO_ROOT,
    generate_batch,
    load_base,
    load_config,
    log_cost,
    resolve_device,
)
from proxies import HEDGE, _question_count, _rate
from steer_sanity import measure_ref_norm

OUT = REPO_ROOT / "results" / "steer_learn"
BASIS = REPO_ROOT / "basis"
PHI_KEYS = ("words", "hedge_per100", "questions_per100")


def load_sl_config():
    with open(REPO_ROOT / "configs" / "steer_learn.yaml") as f:
        return yaml.safe_load(f)


def phi_features(completion):
    """Reward-defining features (trivially computable, no learned models)."""
    words = max(len(completion.split()), 1)
    return {
        "words": float(len(completion.split())),
        "hedge_per100": _rate(completion, HEDGE),
        "questions_per100": 100.0 * _question_count(completion) / words,
    }


def _prompts(n_train, n_test):
    data = json.loads(open(REPO_ROOT / "data" / "prompts.json").read())["train"]
    need = n_train + n_test
    if len(data) < need:
        raise RuntimeError(f"need {need} prompts, data/prompts.json has {len(data)}")
    return data[:n_train], data[n_train:need]


def _gcfg(layer, pcfg):
    return {"steer_layer": layer, "generation": {
        "max_new_tokens": pcfg["max_new_tokens"], "do_sample": True,
        "temperature": pcfg["temperature"], "top_p": pcfg["top_p"]}}


# ---------------------------------------------------------------- phase: pool ----
def phase_pool(base_cfg, slcfg, device, model, tok):
    """Draw n base completions per prompt, compute phi, cache one row per completion."""
    layer = base_cfg["steer_layer"]
    pcfg = slcfg["pool"]
    gcfg = _gcfg(layer, pcfg)
    torch.manual_seed(slcfg["optim"]["seed"])
    train, test = _prompts(pcfg["n_prompts_train"], pcfg["n_prompts_test"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pool.jsonl"
    with open(path, "w") as f:
        for split, prompts in (("train", train), ("test", test)):
            for pi, p in enumerate(prompts):
                comps = generate_batch(model, tok, [p] * pcfg["n_samples"], gcfg)
                for c in comps:
                    if not c.strip():
                        continue
                    f.write(json.dumps({"split": split, "pi": pi, "prompt": p,
                                        "completion": c, "phi": phi_features(c)}) + "\n")
                f.flush()
    print(f"pool -> {path}")


def _load_pool():
    rows = [json.loads(l) for l in open(OUT / "pool.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["pi"]), []).append(r)
    return rows, by


def _reward_fn(rows, slcfg):
    """z-score phi over the TRAIN pool; R = <e*, phi_std>. Returns (R_of_phi, stats)."""
    e = np.array(slcfg["reward"]["target_direction"], dtype=np.float64)
    train_phi = np.array([[r["phi"][k] for k in PHI_KEYS]
                          for r in rows if r["split"] == "train"], dtype=np.float64)
    mu, sd = train_phi.mean(0), train_phi.std(0) + 1e-8
    stats = {"mean": mu.tolist(), "std": sd.tolist(), "e_star": e.tolist()}

    def R(phi):
        v = np.array([phi[k] for k in PHI_KEYS], dtype=np.float64)
        return float(e @ ((v - mu) / sd))
    return R, stats


# -------------------------------------------------- teacher-forced logprob ----
def tf_sum_logprob(model, tok, prefix_ids, comp_ids_list, layer, delta):
    """Sum log p(completion_k | prompt) under injection `delta` (fp32, grad) added to the
    layer-`layer` residual stream at every position. Returns (K,) tensor, grad -> delta."""
    device = model.device
    B, P = len(comp_ids_list), prefix_ids.shape[1]
    lens = [c.shape[1] for c in comp_ids_list]
    Lmax = max(lens)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((B, P + Lmax), pad, dtype=torch.long)
    attn = torch.zeros((B, P + Lmax), dtype=torch.long)
    for k, c in enumerate(comp_ids_list):
        ids[k, :P] = prefix_ids[0]
        ids[k, P:P + lens[k]] = c[0]
        attn[k, :P + lens[k]] = 1
    ids, attn = ids.to(device), attn.to(device)

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + delta.to(hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, attention_mask=attn).logits.float()
    finally:
        h.remove()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for k in range(B):
        L = lens[k]
        idx = torch.arange(P - 1, P - 1 + L, device=device)
        out.append(logp[k, idx, ids[k, P:P + L]].sum())
    return torch.stack(out)


# --------------------------------------------------------------- phase: learn ----
def phase_learn(base_cfg, slcfg, device, model, tok):
    layer = base_cfg["steer_layer"]
    rows, by = _load_pool()
    R, stats = _reward_fn(rows, slcfg)
    train, _ = _prompts(slcfg["pool"]["n_prompts_train"], slcfg["pool"]["n_prompts_test"])
    d = model.config.hidden_size
    ref = measure_ref_norm(model, tok, train[:16], layer)
    cap = slcfg["policy"]["mag_cap_frac"] * ref
    print(f"ref_norm(layer {layer}) = {ref:.1f}; mag cap = {cap:.1f}")

    model.requires_grad_(False)
    r = slcfg["policy"]["rank"]
    torch.manual_seed(slcfg["optim"]["seed"])
    V = torch.nn.Parameter(torch.randn(r, d, device=device) * (cap / (d ** 0.5)))
    coeff = torch.ones(r, device=device)                      # global controller
    opt = torch.optim.Adam([V], lr=slcfg["optim"]["lr"])
    beta, l1, orth = slcfg["reward"]["beta"], slcfg["policy"]["l1"], slcfg["policy"]["orth"]

    # precompute per-prompt: prefix ids, completion ids, tilt weights
    cells = []
    for pi, p in enumerate(train):
        pool = [x for x in by.get(("train", pi), []) if x["completion"].strip()]
        if len(pool) < 2:
            continue
        prefix = tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)["input_ids"]
        comp_ids = [tok(x["completion"], return_tensors="pt",
                        add_special_tokens=False)["input_ids"] for x in pool]
        Rk = torch.tensor([R(x["phi"]) for x in pool])
        w = torch.softmax(Rk / beta, dim=0).to(device)
        cells.append((prefix, comp_ids, w))

    log = []
    for epoch in range(slcfg["optim"]["epochs"]):
        order = torch.randperm(len(cells))
        tot = 0.0
        for i in order.tolist():
            prefix, comp_ids, w = cells[i]
            delta = coeff @ V                                # (d,), grad
            lp = tf_sum_logprob(model, tok, prefix, comp_ids, layer, delta)
            loss = -(w * lp).sum()
            if l1:
                loss = loss + l1 * V.abs().sum()
            if orth and r > 1:
                G = V @ V.t()
                loss = loss + orth * (G - torch.diag(torch.diag(G))).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():                            # project to the mag cap
                nrm = (coeff @ V).norm().item()
                if nrm > cap:
                    V.mul_(cap / nrm)
            tot += float(loss.item())
        print(f"epoch {epoch}: mean loss {tot / len(cells):.3f} | ||delta|| {(coeff@V).norm().item():.1f}")
        log.append(tot / len(cells))

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"V": V.detach().cpu(), "coeff": coeff.cpu(), "ref": ref, "cap": cap,
                "layer": layer, "loss": log}, OUT / "V.pt")
    json.dump(stats, open(OUT / "reward_stats.json", "w"), indent=2)
    print(f"learned V -> {OUT / 'V.pt'}")


# ---------------------------------------------------------------- phase: eval ----
def phase_eval(base_cfg, slcfg, device, model, tok):
    layer = base_cfg["steer_layer"]
    rows, by = _load_pool()
    stats = json.load(open(OUT / "reward_stats.json"))
    mu, sd, e = (np.array(stats[k]) for k in ("mean", "std", "e_star"))

    def R(phi):
        v = np.array([phi[k] for k in PHI_KEYS])
        return float(e @ ((v - mu) / sd))

    ck = torch.load(OUT / "V.pt")
    delta = (ck["coeff"] @ ck["V"]).to(device)
    _, test = _prompts(slcfg["pool"]["n_prompts_train"], slcfg["pool"]["n_prompts_test"])
    gcfg = _gcfg(layer, slcfg["pool"])
    torch.manual_seed(slcfg["optim"]["seed"] + 1)

    base_R, steer_R, base_phi, steer_phi = [], [], [], []
    for pi, p in enumerate(test):
        bpool = [x for x in by.get(("test", pi), []) if x["completion"].strip()]
        if not bpool:
            continue
        base_R.append(np.mean([R(x["phi"]) for x in bpool]))
        base_phi.append(np.mean([[x["phi"][k] for k in PHI_KEYS] for x in bpool], axis=0))
        scomp = generate_batch(model, tok, [p] * slcfg["pool"]["n_samples"], gcfg,
                               vector=delta, alpha=1.0)
        sphi = [phi_features(c) for c in scomp if c.strip()]
        steer_R.append(np.mean([R(ph) for ph in sphi]))
        steer_phi.append(np.mean([[ph[k] for k in PHI_KEYS] for ph in sphi], axis=0))

    base_R, steer_R = np.array(base_R), np.array(steer_R)
    dR = steer_R - base_R
    bphi, sphi = np.array(base_phi).mean(0), np.array(steer_phi).mean(0)

    # best-of-n ceiling on the train pool (context for how much reward is available)
    bo_n = np.mean([max(R(x["phi"]) for x in by[("train", pi)])
                    for pi in range(slcfg["pool"]["n_prompts_train"]) if ("train", pi) in by])

    lines = [
        "# S1.1 — learned reward-driven steering (synthetic positive control)\n",
        f"Target e* over {PHI_KEYS} = {e.tolist()} (z-scored). SmolLM2-1.7B, layer {layer}, "
        f"mag cap {ck['cap']:.0f} (={slcfg['policy']['mag_cap_frac']}·ref {ck['ref']:.0f}), "
        f"rank {slcfg['policy']['rank']} global.\n",
        "## On-policy reward (held-out test prompts)",
        f"- base R:   **{base_R.mean():+.3f}**",
        f"- steered R: **{steer_R.mean():+.3f}**",
        f"- **Δ-R = {dR.mean():+.3f}**  (per-prompt paired; {int((dR>0).sum())}/{len(dR)} prompts up)",
        f"- train-pool best-of-{slcfg['pool']['n_samples']} ceiling: {bo_n:+.3f}",
        "\n## Recovery — realized φ shift (steered − base), should align with e*",
        "| feature | e* | base | steered | Δ | aligned |",
        "|---|---|---|---|---|---|",
    ]
    for j, k in enumerate(PHI_KEYS):
        aligned = "✓" if np.sign(sphi[j] - bphi[j]) == np.sign(e[j]) or e[j] == 0 else "✗"
        lines.append(f"| {k} | {e[j]:+.0f} | {bphi[j]:.2f} | {sphi[j]:.2f} | {sphi[j]-bphi[j]:+.2f} | {aligned} |")

    # optional: cosine to the contrastive axes, if extracted
    try:
        axes = np.load(BASIS / "axes.npz")
        v = (ck["coeff"] @ ck["V"]).numpy()
        v = v / (np.linalg.norm(v) + 1e-8)
        cos = {n.split("|")[0]: float(v @ (axes[n] / (np.linalg.norm(axes[n]) + 1e-8)))
               for n in axes.files if n.endswith(f"|{layer}")}
        lines.append("\n## Cosine of learned direction to contrastive axes")
        lines += [f"- {n}: {c:+.2f}" for n, c in sorted(cos.items(), key=lambda t: -abs(t[1]))]
    except FileNotFoundError:
        pass

    lines += ["\n## Reading",
              "GREEN-for-S1.1: Δ-R > 0 on held-out prompts AND φ moves toward e* — the RWR "
              "machinery learns a reward-aligned steering direction from scratch. This is the "
              "machinery check before the real RM (S1.2)."]
    (BASIS).mkdir(exist_ok=True)
    (BASIS / "s1_synth_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if not l.startswith("|")))
    print(f"\nreport -> {BASIS / 's1_synth_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pool", "learn", "eval", "all"], default="all")
    args = ap.parse_args()
    base_cfg = load_config()
    slcfg = load_sl_config()
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    if args.phase in ("pool", "all"):
        phase_pool(base_cfg, slcfg, device, model, tok)
    if args.phase in ("learn", "all"):
        phase_learn(base_cfg, slcfg, device, model, tok)
    if args.phase in ("eval", "all"):
        phase_eval(base_cfg, slcfg, device, model, tok)
    print(log_cost("S1", f"steer_learn_{args.phase}", time.time() - t0, device,
                   notes="synthetic positive control"))


if __name__ == "__main__":
    main()
