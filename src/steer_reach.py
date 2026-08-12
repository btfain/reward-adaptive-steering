"""Subproject 1 / S1.2 — per-prompt free-δ REACHABILITY probe (rank fully relaxed).

The decisive question the fixed-basis arms could not answer: does a reward-INCREASING,
fluency-preserving steering direction EXIST for the real RM at all? Here we drop every
structural assumption (no shared basis, no controller, no low rank) and, for each prompt,
learn an unconstrained full-rank δ ∈ R^d with our own method (RWR toward the KL-tilt on that
prompt's pool), at the fluent caps. Deliberate per-prompt overfitting — we want the ceiling.

Instrumented to separate the live hypotheses:
  headline    on-policy ΔRM as a fraction of the best-of-n ceiling. ~0 fluent => reachability
              WALL (steering fundamentally bounded here); large => reachable dirs EXIST and our
              global/low-rank/controller failures were LEARNING/design, not reachability.
  surrogate   per-prompt ΔL of the RWR objective (did it optimize? guards the 'undertrained'
              escape hatch). Optimizes but on-policy flat => TF-surrogate/on-policy GAP.
  guard       distinct-2 + base-NLL drift (anti-collapse, rule 4).
  factor      (conditional on headroom) effective rank of the per-prompt optima + their
              predictability from h(x) — does a low-rank basis + controller reconstruct them?
              Low rank + high h->δ R² => S1.2's joint-learning failure was OPTIMIZATION, and a
              staged 'solve-per-prompt-then-factor' fix is viable. High rank / unpredictable =>
              the low-rank+controller design was mis-specified.

    python src/steer_reach.py --base-config configs/base_7b.yaml --config configs/steer_reach_7b.yaml
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
from steer_cond import _prompts, read_state
from steer_learn import _boot, tf_sum_logprob
from steer_sanity import measure_ref_norm
from steer_sweep import _base_nll, _distinct2, _prefix_ids

OUT = REPO_ROOT / "results" / "steer_reach"
BASIS = REPO_ROOT / "basis"
REPORT = "s1_reach_report.md"


def load_reach_config(path=None):
    with open(path or (REPO_ROOT / "configs" / "steer_reach.yaml")) as f:
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
    return by


def _fit_delta(model, tok, prefix, comp_ids, w, layer, d, cap, steps, lr, mag_pen, device):
    """Learn an unconstrained full-rank δ for ONE prompt by RWR on its pool. Returns
    (δ_at_cap, L0, Lf) — the projected-to-cap direction and the surrogate loss endpoints."""
    delta = torch.zeros(d, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    L0 = None
    for _ in range(steps):
        lp = tf_sum_logprob(model, tok, prefix, comp_ids, layer, delta)
        loss = -(w * lp).sum() + mag_pen * torch.relu(delta.norm() - cap) ** 2
        if L0 is None:
            L0 = float(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    Lf = float(loss.item())
    with torch.no_grad():
        n = delta.norm()
        dcap = delta * (cap / n) if n > 0 else delta
    return dcap.detach(), L0, Lf


def _onpolicy_drm(model, tok, gcfg, prompt, delta, base_r, rm, rm_tok, m):
    sc = [c for c in generate_batch(model, tok, [prompt] * m, gcfg, vector=delta, alpha=1.0) if c.strip()]
    if not sc:
        return None, []
    return float(np.mean([rm_score(rm, rm_tok, prompt, c) for c in sc])) - base_r, sc


def _factorize(D, H, is_train, ridge, n_pca):
    """Effective rank of the per-prompt optima + predictability of δ from h(x).
    D:(N,d) projected deltas, H:(N,d) read states, is_train:(N,) bool split mask."""
    Dc = D - D.mean(0)
    s = np.linalg.svd(Dc, compute_uv=False)
    var = s ** 2; cum = np.cumsum(var) / var.sum()
    r50 = int(np.searchsorted(cum, 0.50) + 1)
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    # PCA-reduce h (fit on train) then ridge h_pca -> δ; honest train/test R² and cosine
    Htr = H[is_train]; mu = Htr.mean(0)
    _, _, Vh = np.linalg.svd(Htr - mu, full_matrices=False)
    Wp = Vh[:n_pca].T                                  # (d, n_pca)
    Ztr = (Htr - mu) @ Wp
    Zte = (H[~is_train] - mu) @ Wp
    Dtr, Dte = D[is_train], D[~is_train]
    A = Ztr.T @ Ztr + ridge * np.eye(Ztr.shape[1])
    Wmap = np.linalg.solve(A, Ztr.T @ Dtr)             # (n_pca, d)
    pred = Zte @ Wmap
    ss_res = ((Dte - pred) ** 2).sum()
    ss_tot = ((Dte - Dtr.mean(0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    cos = float(np.mean([
        (Dte[i] @ pred[i]) / (np.linalg.norm(Dte[i]) * np.linalg.norm(pred[i]) + 1e-9)
        for i in range(len(Dte))]))
    return {"r50": r50, "r90": r90, "n_dirs": int(D.shape[0]), "probe_r2": r2, "probe_cos": cos}


def phase_reach(base_cfg, rcfg, device, model, tok, rm, rm_tok):
    layer = base_cfg["steer_layer"]
    read_layer = rcfg["read_layer"]
    by = _load_pool(REPO_ROOT / rcfg["pool_dir"])
    P = _prompts(rcfg["pool"]["n_prompts_train"], rcfg["pool"]["n_prompts_test"])
    d = model.config.hidden_size
    ref = measure_ref_norm(model, tok, P["train"][:16], layer)
    gcfg = _gcfg(layer, rcfg["pool"])
    m = rcfg["eval"]["n_samples"]
    steps, lr = rcfg["inner"]["steps"], rcfg["inner"]["lr"]
    mag_pen = rcfg["inner"]["mag_penalty"]
    fac_frac = rcfg["factor"]["cap_frac"]
    model.requires_grad_(False)
    torch.manual_seed(rcfg["optim"]["seed"] + 7)
    print(f"ref_norm(layer {layer}) = {ref:.1f}")

    items = [("train", pi, p) for pi, p in enumerate(P["train"])] + \
            [("test", pi, p) for pi, p in enumerate(P["test"])]
    fracs = rcfg["mag_fracs"]

    # best-of-n ceiling (RM1) on the test split, for the headroom denominator
    bo = []
    for pi, _ in enumerate(P["test"]):
        r1 = [x["rm"] for x in by.get(("test", pi), []) if x["completion"].strip()]
        if r1:
            bo.append(max(r1) - np.mean(r1))
    bo_n = float(np.mean(bo))
    base_d2 = _distinct2([x["completion"] for pi, _ in enumerate(P["test"])
                          for x in by.get(("test", pi), []) if x["completion"].strip()])

    OUT.mkdir(parents=True, exist_ok=True)
    per = {f: {"drm": [], "L0": [], "Lf": [], "texts": [], "split": []} for f in fracs}
    Dfac, Hfac, split_fac = [], [], []   # per-prompt δ at fac_frac, read states, split mask
    guard_nll_steer, guard_nll_base = [], []

    for idx, (split, pi, prompt) in enumerate(items):
        pool = [x for x in by.get((split, pi), []) if x["completion"].strip()]
        if len(pool) < 2:
            continue
        h = read_state(model, tok, prompt, read_layer).to(device)
        prefix = _prefix_ids(tok, prompt)
        comp_ids = [tok(x["completion"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                    for x in pool]
        Rk = torch.tensor([x["rm"] for x in pool])
        w = torch.softmax(Rk / rcfg["reward"]["beta"], dim=0).to(device)
        base_r = float(np.mean([x["rm"] for x in pool]))
        for f in fracs:
            cap = f * ref
            dcap, L0, Lf = _fit_delta(model, tok, prefix, comp_ids, w, layer, d, cap,
                                      steps, lr, mag_pen, device)
            drm, sc = _onpolicy_drm(model, tok, gcfg, prompt, dcap, base_r, rm, rm_tok, m)
            if drm is None:
                continue
            per[f]["drm"].append(drm); per[f]["L0"].append(L0); per[f]["Lf"].append(Lf)
            per[f]["texts"] += sc; per[f]["split"].append(split)
            if abs(f - fac_frac) < 1e-9:
                Dfac.append(dcap.float().cpu().numpy()); Hfac.append(h.float().cpu().numpy())
                split_fac.append(split == "train")
                if len(guard_nll_steer) < rcfg["eval"]["guard_prompts"] and sc:
                    ns = _base_nll(model, tok, layer, d, prompt, sc[0], device)
                    nb = _base_nll(model, tok, layer, d, prompt, pool[0]["completion"], device)
                    if ns is not None and nb is not None:
                        guard_nll_steer.append(ns); guard_nll_base.append(nb)
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(items)} prompts")

    fac = _factorize(np.stack(Dfac), np.stack(Hfac), np.array(split_fac),
                     rcfg["factor"]["ridge"], rcfg["factor"]["n_pca"]) if Dfac else None
    np.savez(OUT / "deltas.npz", D=np.stack(Dfac), H=np.stack(Hfac),
             split=np.array(split_fac), ref=ref, fac_frac=fac_frac)
    _write_report(base_cfg, rcfg, layer, read_layer, ref, bo_n, base_d2, per, fac,
                  float(np.mean(guard_nll_steer)) if guard_nll_steer else float("nan"),
                  float(np.mean(guard_nll_base)) if guard_nll_base else float("nan"))


def _write_report(base_cfg, rcfg, layer, read_layer, ref, bo_n, base_d2, per, fac, nll_s, nll_b):
    mname = base_cfg["base_model"].split("/")[-1]
    m = rcfg["eval"]["n_samples"]
    L = ["# S1.2 — per-prompt free-δ reachability probe (rank relaxed)\n",
         f"{mname}, steer L{layer}, read L{read_layer}. Per prompt: unconstrained full-rank δ∈R^d "
         f"learned by RWR on its own pool ({rcfg['pool_dir']}), {rcfg['inner']['steps']} steps, "
         f"evaluated on-policy at exactly the cap. ref_norm={ref:.0f}. RM1="
         f"{base_cfg['reward_model'].split('/')[-1]}, m={m} samples.\n",
         f"Denominator: best-of-n ceiling **{bo_n:+.2f}** (RM1); prompting ref +1.08; A2 contrastive ~0. "
         f"Base distinct-2 = {base_d2:.3f}.\n",
         "| frac | cap | on-policy ΔRM1 [95% CI] | frac of best-of-n | surrogate ΔL (mean) | distinct-2 |",
         "|---|---|---|---|---|---|"]
    for f in rcfg["mag_fracs"]:
        drm = np.array(per[f]["drm"]); lo, hi = _boot(drm)
        dL = np.mean(np.array(per[f]["L0"]) - np.array(per[f]["Lf"]))
        L.append(f"| {f} | {f * ref:.0f} | {drm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                 f"{drm.mean() / bo_n:+.2f} | {dL:+.2f} | {_distinct2(per[f]['texts']):.2f} |")
    L += ["", f"Fluency guard at frac {rcfg['factor']['cap_frac']}: base-NLL steered {nll_s:.2f} "
          f"vs base {nll_b:.2f} (spike ⇒ gains bought with fluency).", ""]
    if fac:
        L += ["## Factorization of the per-prompt optima (conditional on headroom existing)",
              f"- Effective rank of the {fac['n_dirs']} per-prompt δ's: **{fac['r50']}** dirs for 50% "
              f"variance, **{fac['r90']}** for 90% (of {base_cfg.get('hidden_size','d')}-dim).",
              f"- Predictability from h(x) (ridge, held-out): **R²={fac['probe_r2']:+.2f}**, "
              f"mean cosine **{fac['probe_cos']:+.2f}**.",
              "- Low rank + high R²/cos ⇒ a basis+controller could reconstruct these ⇒ S1.2's joint "
              "failure was OPTIMIZATION (staged solve-then-factor is viable). High rank / low R² ⇒ the "
              "low-rank+controller design was mis-specified.", ""]
    L += ["## Reading key",
          "- **on-policy ΔRM ≈ 0 while fluent** ⇒ reachability WALL: even per-prompt full-rank fluent "
          "steering can't move the RM ⇒ steering fundamentally bounded here ⇒ pivot with a clean null.",
          "- **ΔRM a large fraction of best-of-n** ⇒ reachable reward-increasing dirs EXIST ⇒ our "
          "global/low-rank/controller failures were LEARNING/design, not reachability ⇒ worth improving.",
          "- **surrogate ΔL large but ΔRM ≈ 0** ⇒ TF-surrogate optimized but doesn't transfer on-policy "
          "⇒ objective/on-policy GAP (motivates on-policy RL), not a reachability wall.",
          "- Judge against best-of-n and prompting (+1.08); watch the fluency guard for collapse."]
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
    rcfg = load_reach_config(args.config)
    tag = rcfg.get("tag", "")
    if tag:
        OUT = REPO_ROOT / "results" / f"steer_reach_{tag}"
        REPORT = f"s1_reach_{tag}_report.md"
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    rm, rm_tok = load_rm(base_cfg, device)  # RM1 only (RM2 would OOM the autograd-heavy inner loop)
    phase_reach(base_cfg, rcfg, device, model, tok, rm, rm_tok)
    print(log_cost("S1", "steer_reach", time.time() - t0, device, notes="per-prompt reachability probe"))


if __name__ == "__main__":
    main()
