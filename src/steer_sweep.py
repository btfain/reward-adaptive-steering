"""Subproject 1 / S1.2 diagnostic sweep — characterize the steering ceiling and decide
whether the real-RM null is an ACTION-SPACE limit (steering can't move the RM at fluent
magnitude) or a LEARNING limit (good fluent directions exist but we fail to find/generalize).

Reuses the already-generated 7B pool (results/steer_rm_7b/pool.jsonl by default) — no
regeneration. For each magnitude cap in {0.05,0.10,0.15,0.20}*ref_norm it evaluates, held-out:

  learned-global  RWR-learned single direction, reported on TRAIN and HELD-OUT
                  (fit-vs-generalize split — the learning-side discriminator).
  contrastive     non-learned CAA direction (mean high-reward act − mean low-reward act);
                  a reward-informed direction that involves NO gradient learning.
  oracle          per-prompt best-of-K over a reward-relevant dictionary (contrastive +
                  random directions in the top-r PCA subspace of pool activations); a
                  direction-agnostic CEILING on what any fluent-magnitude steering could do.
  guard           distinct-2 (diversity) and base-model NLL drift of the steered text
                  (fluency/collapse guard, per CLAUDE.md rule 4).

Plus a capacity side-check: linear conditional at the fluent cap, rank 8.

Reading key:
  oracle ≈ 0 across the fluent band            -> ACTION SPACE empty: steering can't help
                                                   at fluent magnitude; learning irrelevant.
  oracle high, learned-heldout low, train≈heldout low -> good fluent dirs EXIST, not capturable.
  learned-train high, learned-heldout low      -> GENERALIZATION problem specifically.
  everything only rises where the guard spikes -> confirms the fluency/leverage vise.

    python src/steer_sweep.py --base-config configs/base_7b.yaml --config configs/steer_sweep_7b.yaml
"""

import argparse
import json
import time

import numpy as np
import torch
import yaml
from models import (
    REPO_ROOT, generate_batch, load_base, load_config, load_rm, log_cost,
    resolve_device, rm_score,
)
from steer_cond import Policy, _prompts, read_state
from steer_learn import _boot, tf_sum_logprob
from steer_rm import _train_arm
from steer_sanity import measure_ref_norm

OUT = REPO_ROOT / "results" / "steer_sweep"
BASIS = REPO_ROOT / "basis"
REPORT = "s1_sweep_report.md"


def load_sweep_config(path=None):
    with open(path or (REPO_ROOT / "configs" / "steer_sweep.yaml")) as f:
        return yaml.safe_load(f)


def _gcfg(layer, pcfg):
    return {"steer_layer": layer, "generation": {
        "max_new_tokens": pcfg["max_new_tokens"], "do_sample": True,
        "temperature": pcfg["temperature"], "top_p": pcfg["top_p"]}}


def _load_pool(pool_dir):
    rows = [json.loads(l) for l in open(pool_dir / "pool.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["pi"]), []).append(r)
    return rows, by


def _prefix_ids(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)["input_ids"]


def _completion_act(model, tok, prompt, completion, layer, device):
    """Mean steer-layer hidden state over the completion token positions (frozen base)."""
    pids = _prefix_ids(tok, prompt).to(device)
    cids = tok(completion, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    ids = torch.cat([pids, cids], dim=1)
    cap = {}
    def hook(m, i, o):
        cap["h"] = o[0] if isinstance(o, tuple) else o
    hh = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        hh.remove()
    return cap["h"][0, pids.shape[1]:, :].mean(0).float()  # (d,) — float32 for SVD (no bf16 svd kernel)


def _distinct2(texts):
    """Fraction of distinct word bigrams across a list of texts (diversity guard)."""
    bg, tot = set(), 0
    for t in texts:
        w = t.split()
        for a, b in zip(w, w[1:]):
            bg.add((a, b)); tot += 1
    return len(bg) / tot if tot else 0.0


def _base_nll(model, tok, layer, d, prompt, completion, device):
    """Per-token NLL of a completion under the FROZEN base (delta=0). Higher => more off-distribution."""
    prefix = _prefix_ids(tok, prompt)
    cids = tok(completion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if cids.shape[1] == 0:
        return None
    zero = torch.zeros(d, device=device)
    with torch.no_grad():
        lp = tf_sum_logprob(model, tok, prefix, [cids], layer, zero)[0]
    return float(-lp / cids.shape[1])


# ---- direction dictionaries (built from TRAIN pool only — honest about generalization) ----
def _build_dictionary(model, tok, by, P, layer, device, r_sub, k_rand, seed):
    """Return (contrastive_unit, [random units in top-r PCA subspace of pool acts])."""
    acts, rews = [], []
    for pi, prompt in enumerate(P["train"]):
        for x in by.get(("train", pi), []):
            if x["completion"].strip():
                acts.append(_completion_act(model, tok, prompt, x["completion"], layer, device))
                rews.append(x["rm"])
    A = torch.stack(acts).float()              # (N, d) — float32 (bf16 has no svd/precise-mean kernel)
    R = torch.tensor(rews, device=device)
    hi, lo = R >= R.median(), R < R.median()
    vc = (A[hi].mean(0) - A[lo].mean(0))
    vc = vc / (vc.norm() + 1e-9)               # contrastive (CAA) unit direction
    _, _, Vh = torch.linalg.svd(A - A.mean(0), full_matrices=False)
    U = Vh[:r_sub]                             # (r_sub, d) reward-relevant subspace
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = []
    for _ in range(k_rand):
        c = torch.randn(r_sub, generator=g).to(device)
        v = c @ U
        rand.append(v / (v.norm() + 1e-9))
    return vc, rand


def _arm_delta_rm(model, tok, gcfg, prompts_state, base1, prompt_of, unit, cap, rm, rm_tok, m, device):
    """Held-out ΔRM for a single fixed unit direction scaled to exactly `cap` (all fixed-direction
    arms share the same magnitude, isolating direction quality). Returns (array, texts, pis) where
    pis is the aligned list of prompt-ids actually scored (skips prompts with no valid generation)."""
    delta = (unit * cap).to(device)
    ds, texts, pis = [], [], []
    for pi in prompts_state:
        sc = [c for c in generate_batch(model, tok, [prompt_of[pi]] * m, gcfg,
                                        vector=delta, alpha=1.0) if c.strip()]
        if not sc:
            continue
        texts += sc
        ds.append(np.mean([rm_score(rm, rm_tok, prompt_of[pi], c) for c in sc]) - base1[pi])
        pis.append(pi)
    return np.array(ds), texts, pis


def phase_sweep(base_cfg, scfg, device, model, tok, rm, rm_tok):
    layer = base_cfg["steer_layer"]
    read_layer = scfg["policy"]["read_layer"]
    pool_dir = REPO_ROOT / scfg["pool_dir"]
    rows, by = _load_pool(pool_dir)
    P = _prompts(scfg["pool"]["n_prompts_train"], scfg["pool"]["n_prompts_test"])
    d = model.config.hidden_size
    ref = measure_ref_norm(model, tok, P["train"][:16], layer)
    gcfg = _gcfg(layer, scfg["pool"])
    m = scfg["eval"]["n_samples"]
    m_or = scfg["eval"]["oracle_samples"]
    n_tr = scfg["eval"]["n_train_eval"]
    model.requires_grad_(False)
    torch.manual_seed(scfg["optim"]["seed"] + 7)
    print(f"ref_norm(layer {layer}) = {ref:.1f}")

    # ---- training cells (cap-independent) for the learned arms ----
    def build_cells(split, prompts):
        cells = []
        for pi, prompt in enumerate(prompts):
            pool = [x for x in by.get((split, pi), []) if x["completion"].strip()]
            if len(pool) < 2:
                continue
            h = read_state(model, tok, prompt, read_layer).to(device)
            prefix = _prefix_ids(tok, prompt)
            comp_ids = [tok(x["completion"], return_tensors="pt",
                            add_special_tokens=False)["input_ids"] for x in pool]
            Rk = torch.tensor([x["rm"] for x in pool])
            w = torch.softmax(Rk / scfg["reward"]["beta"], dim=0).to(device)
            cells.append((h, prefix, comp_ids, w, pi))
        return cells
    tr_cells = build_cells("train", P["train"])
    train_cells = [(h, pre, ci, w) for (h, pre, ci, w, _) in tr_cells]

    # ---- held-out prompt states + base RM baseline + best-of-n ceiling ----
    # RM2 is intentionally NOT loaded here: it would sit resident during the autograd-heavy
    # training and OOM the 24GB card. steer_rm already established RM1/RM2 agreement at 7B; the
    # sweep's own guard (distinct-2 + base-NLL drift) is the relevant anti-hacking cross-check.
    tstate, base1, prompt_of, bo = {}, {}, {}, []
    for pi, prompt in enumerate(P["test"]):
        bpool = [x for x in by.get(("test", pi), []) if x["completion"].strip()]
        if not bpool:
            continue
        tstate[pi] = read_state(model, tok, prompt, read_layer).to(device)
        prompt_of[pi] = prompt
        r1 = [x["rm"] for x in bpool]
        base1[pi] = float(np.mean(r1))
        bo.append(max(r1) - np.mean(r1))
    bo_n = float(np.mean(bo))
    base_d2 = _distinct2([x["completion"] for pi in tstate
                          for x in by.get(("test", pi), []) if x["completion"].strip()])
    # train-subset baselines (RM1 only) for the fit metric
    tr_state, tr_base1, tr_prompt = {}, {}, {}
    for pi, prompt in list(enumerate(P["train"]))[:n_tr]:
        bpool = [x for x in by.get(("train", pi), []) if x["completion"].strip()]
        if not bpool:
            continue
        tr_state[pi] = read_state(model, tok, prompt, read_layer).to(device)
        tr_prompt[pi] = prompt
        tr_base1[pi] = float(np.mean([x["rm"] for x in bpool]))

    # ---- reward-relevant direction dictionary for the oracle (TRAIN only) ----
    vc, rand_dirs = _build_dictionary(model, tok, by, P, layer, device,
                                      scfg["oracle"]["r_subspace"], scfg["oracle"]["k_random"],
                                      scfg["optim"]["seed"])
    cand_dirs = [vc] + rand_dirs

    OUT.mkdir(parents=True, exist_ok=True)
    fracs = scfg["mag_fracs"]
    grid = {}
    for frac in fracs:
        cap = frac * ref
        row = {"cap": cap}
        print(f"\n=== magnitude frac {frac} (cap {cap:.1f}) ===")

        # learned-global: train it at THIS cap, eval on held-out and on a train subset
        rvar = {"policy": {**scfg["policy"], "rank": 1, "arms": ["global"]}, "optim": scfg["optim"]}
        pol = _train_arm("global", train_cells, d, cap, rvar, layer, model, tok)
        pol.eval()
        gunit = None
        with torch.no_grad():
            gd = pol.delta(next(iter(tstate.values())))
            gunit = (gd / (gd.norm() + 1e-9)).detach()   # global dir is prompt-independent

        d1, ltexts, _ = _arm_delta_rm(model, tok, gcfg, tstate, base1, prompt_of, gunit, cap, rm, rm_tok, m, device)
        tr1, _, _ = _arm_delta_rm(model, tok, gcfg, tr_state, tr_base1, tr_prompt, gunit, cap, rm, rm_tok, m, device)
        lo1, hi1 = _boot(d1)
        # fluency drift of the learned-global steered text (first sample per prompt)
        nlls = [n for pi in list(tstate)[:scfg["eval"]["guard_prompts"]]
                for c in [next((c for c in generate_batch(model, tok, [prompt_of[pi]], gcfg,
                                vector=(gunit * cap).to(device), alpha=1.0) if c.strip()), None)]
                if c is not None for n in [_base_nll(model, tok, layer, d, prompt_of[pi], c, device)]
                if n is not None]
        base_nll = [n for pi in list(tstate)[:scfg["eval"]["guard_prompts"]]
                    for x in by.get(("test", pi), [])[:1] if x["completion"].strip()
                    for n in [_base_nll(model, tok, layer, d, prompt_of[pi], x["completion"], device)]
                    if n is not None]
        row["learned"] = {"tr": float(tr1.mean()), "d1": float(d1.mean()), "lo1": lo1, "hi1": hi1,
                          "d2div": _distinct2(ltexts),
                          "nll": float(np.mean(nlls)) if nlls else float("nan"),
                          "nll_base": float(np.mean(base_nll)) if base_nll else float("nan")}

        # contrastive (non-learned)
        c1, ctexts, _ = _arm_delta_rm(model, tok, gcfg, tstate, base1, prompt_of, vc, cap, rm, rm_tok, m, device)
        clo, chi = _boot(c1)
        row["contrastive"] = {"d1": float(c1.mean()), "lo": clo, "hi": chi, "d2div": _distinct2(ctexts)}

        # oracle: per-prompt best-of-K over the reward-relevant dictionary
        per_dir = {}   # dir_idx -> {pi: mean ΔRM1}  (aligned by the pis _arm_delta_rm actually scored)
        otexts = []
        for j, u in enumerate(cand_dirs):
            arr, tx, pis = _arm_delta_rm(model, tok, gcfg, tstate, base1, prompt_of, u, cap, rm, rm_tok, m_or, device)
            otexts += tx
            per_dir[j] = {pi: v for pi, v in zip(pis, arr)}
        best = []
        for pi in tstate:
            vals = [per_dir[j][pi] for j in per_dir if pi in per_dir[j]]
            if vals:
                best.append(max(vals))
        best = np.array(best)
        olo, ohi = _boot(best)
        row["oracle"] = {"d1": float(best.mean()), "lo": olo, "hi": ohi, "d2div": _distinct2(otexts)}
        grid[frac] = row
        torch.save({"global_unit": gunit.cpu(), "cap": cap}, OUT / f"global_frac{frac}.pt")

    # ---- capacity side-check: linear conditional at the fluent cap, rank 8 ----
    side = {}
    fcap = scfg["side_check"]["frac"] * ref
    for rk in scfg["side_check"]["ranks"]:
        rvar = {"policy": {**scfg["policy"], "rank": rk, "arms": ["linear"]}, "optim": scfg["optim"]}
        pol = _train_arm("linear", train_cells, d, fcap, rvar, layer, model, tok)
        pol.eval()
        d1 = []
        for pi in tstate:
            with torch.no_grad():
                delta = pol.delta(tstate[pi])
                dn = delta.norm()
                if dn > fcap:
                    delta = delta * (fcap / dn)
            sc = [c for c in generate_batch(model, tok, [prompt_of[pi]] * m, gcfg,
                                            vector=delta, alpha=1.0) if c.strip()]
            if sc:
                d1.append(np.mean([rm_score(rm, rm_tok, prompt_of[pi], c) for c in sc]) - base1[pi])
        d1 = np.array(d1); lo, hi = _boot(d1)
        side[rk] = {"d1": float(d1.mean()), "lo": lo, "hi": hi}

    _write_report(base_cfg, scfg, layer, read_layer, ref, bo_n, base_d2, grid, side, fcap)
    json.dump({"ref": ref, "grid": {str(k): v for k, v in grid.items()}, "side": side},
              open(OUT / "sweep.json", "w"), indent=2)


def _write_report(base_cfg, scfg, layer, read_layer, ref, bo_n, base_d2, grid, side, fcap):
    mname = base_cfg["base_model"].split("/")[-1]
    m = scfg["eval"]["n_samples"]
    L = ["# S1.2 diagnostic sweep — steering ceiling: action-space limit vs learning limit\n",
         f"{mname}, steer L{layer}, read L{read_layer}. Reuses the S1.2 real-RM pool "
         f"({scfg['pool_dir']}). ref_norm={ref:.0f}; caps are frac×ref. Held-out ΔRM1 paired vs base, "
         f"m={m} samples (oracle m={scfg['eval']['oracle_samples']}). RM1="
         f"{base_cfg['reward_model'].split('/')[-1]}. **All fixed directions (learned/contrastive/oracle) "
         f"injected at exactly the cap magnitude** — the favorable case, isolating direction quality "
         f"from magnitude (so learned here ≥ steer_rm's raw-magnitude global by construction).\n",
         f"Reference (A2, 7B): contrastive ~0 (+0.15); prompting +1.08; best-of-n ceiling "
         f"**{bo_n:+.2f}** (RM1). Base distinct-2 = {base_d2:.3f}. RM2 dropped here (would OOM training); "
         f"RM1/RM2 agreement already established in the steer_rm run.\n",
         "| frac | cap | learned train ΔRM1 | learned heldout ΔRM1 [95% CI] | contrastive ΔRM1 | **oracle ΔRM1** [CI] | distinct-2 L/O (base {:.2f}) | base-NLL steer/base |".format(base_d2),
         "|---|---|---|---|---|---|---|---|"]
    for frac in scfg["mag_fracs"]:
        r = grid[frac]; le = r["learned"]; c = r["contrastive"]; o = r["oracle"]
        L.append(
            f"| {frac} | {r['cap']:.0f} | {le['tr']:+.3f} | {le['d1']:+.3f} [{le['lo1']:+.3f}, {le['hi1']:+.3f}] | "
            f"{c['d1']:+.3f} | "
            f"**{o['d1']:+.3f}** [{o['lo']:+.3f}, {o['hi']:+.3f}] | {le['d2div']:.2f}/{o['d2div']:.2f} | "
            f"{le['nll']:.2f}/{le['nll_base']:.2f} |")
    L += ["", "## Capacity side-check — linear conditional at frac "
          f"{scfg['side_check']['frac']} (cap {fcap:.0f})",
          "| rank | heldout ΔRM1 [95% CI] |", "|---|---|"]
    for rk in scfg["side_check"]["ranks"]:
        s = side[rk]
        L.append(f"| {rk} | {s['d1']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] |")
    L += ["", "## Reading key",
          "- **oracle ≈ 0 across the fluent band** ⇒ ACTION SPACE empty: no fluent reward-increasing "
          "direction exists in the reward-relevant subspace; learning is not the bottleneck.",
          "- **oracle high but learned-heldout low, train≈heldout low** ⇒ good fluent directions EXIST "
          "but are not capturable by a fixed/learned policy ⇒ LEARNING/structure limit.",
          "- **learned-train high but learned-heldout low** ⇒ GENERALIZATION limit specifically.",
          "- **everything only rises where distinct-2 collapses / base-NLL spikes** ⇒ the fluency vise: "
          "reward gains are bought with fluency, not fluent steering (anti-collapse guard, rule 4).",
          "- Judge every number against best-of-n above and prompting (+1.08)."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / REPORT).write_text("\n".join(L) + "\n")
    print("\n".join(x for x in L if not x.startswith("|")))
    print(f"\nreport -> {BASIS / REPORT}")


def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-config", default=None)
    args = ap.parse_args()
    base_cfg = load_config(args.base_config)
    scfg = load_sweep_config(args.config)
    tag = scfg.get("tag", "")
    if tag:
        OUT = REPO_ROOT / "results" / f"steer_sweep_{tag}"
        REPORT = f"s1_sweep_{tag}_report.md"
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    rm, rm_tok = load_rm(base_cfg, device)  # RM1 only; RM2 omitted to keep training within 24GB
    phase_sweep(base_cfg, scfg, device, model, tok, rm, rm_tok)
    print(log_cost("S1", "steer_sweep", time.time() - t0, device, notes="steering-ceiling diagnostic sweep"))


if __name__ == "__main__":
    main()
