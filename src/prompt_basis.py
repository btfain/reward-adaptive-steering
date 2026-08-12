"""Subproject 1 (revised method) — procedural-prompt basis + learned router.

The steering method hit a coverage wall (the reachable reward directions don't compress to a
small basis). This pivots the SAME reward-adaptive idea into the action space that A2 showed
actually moves reward — natural procedural prompt-moves — and makes basis DISCOVERY a principled
optimization rather than a handpick. Three subprocedures:

  (1) initialize X   ABSTRACTED here: X is a curated candidate file (configs/candidates_seed.txt),
                     an opaque input to the algorithm. (Real init = LLM-from-preferences + verify
                     + dedup, built separately.)
  (2) select basis   greedy SUBMODULAR selection: choose <=K moves maximizing
                     f(S)=Σ_x max(0, max_{p∈S} swing(x,p)), swing(x,p)=RM(with p)−RM(base).
                     Monotone submodular => greedy is (1−1/e)-optimal. Emits a value-vs-K curve.
  (3) learn router   a K-way classifier h(x)->move (+ 'none'), read-layer SWEPT and chosen by
                     held-out routing accuracy; linear and MLP. No backprop through the LLM — the
                     router trains on cached features, which is also the lean-vs-LoRA cost story.

Load-bearing test (held-out): does the router beat the best SINGLE unconditional move (is
conditioning worth it?), and how close to the oracle-over-basis ceiling; vs A2 prompting +1.08,
best-of-n +1.40. Reuses results/steer_rm_7b/pool.jsonl for base(x). RM1 only; no autograd → lean.

    python src/prompt_basis.py --phase all --base-config configs/base_7b.yaml --config configs/prompt_basis_7b.yaml
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models import (
    REPO_ROOT, generate_batch, load_base, load_config, load_rm, log_cost,
    resolve_device, rm_score,
)
from steer_cond import _prompts
from steer_learn import _boot
from steer_sweep import _distinct2

OUT = REPO_ROOT / "results" / "prompt_basis"
BASIS = REPO_ROOT / "basis"
REPORT = "s1_pbasis_report.md"


def load_pb_config(path=None):
    with open(path or (REPO_ROOT / "configs" / "prompt_basis.yaml")) as f:
        return yaml.safe_load(f)


def _read_candidates(path):
    out = []
    for line in open(REPO_ROOT / path):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _gcfg(steer_layer, pcfg):
    return {"steer_layer": steer_layer, "generation": {
        "max_new_tokens": pcfg["max_new_tokens"], "do_sample": True,
        "temperature": pcfg["temperature"], "top_p": pcfg["top_p"]}}


def _base_by_prompt(pool_dir):
    """Mean base RM per (split, pi) from the reused pool; also the base completions for the guard."""
    rows = [json.loads(l) for l in open(pool_dir / "pool.jsonl")]
    r, texts = {}, {}
    for x in rows:
        if x["completion"].strip():
            r.setdefault((x["split"], x["pi"]), []).append(x["rm"])
            texts.setdefault((x["split"], x["pi"]), []).append(x["completion"])
    base = {k: float(np.mean(v)) for k, v in r.items()}
    return base, texts


@torch.no_grad()
def _read_states_multi(model, tok, prompt, layers):
    """Last-token residual state at several layers in ONE forward pass (fp32, cpu)."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    cap, handles = {}, []
    for L in layers:
        handles.append(model.model.layers[L].register_forward_hook(
            lambda _m, _i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())))
    try:
        model(enc["input_ids"])
    finally:
        for h in handles:
            h.remove()
    return {L: cap[L][0, -1, :].float().cpu().numpy() for L in layers}


# ---------------------------------------------------------------- phase: swing ----
def phase_swing(base_cfg, pb, device, model, tok, rm, rm_tok):
    steer_layer = base_cfg["steer_layer"]
    gcfg = _gcfg(steer_layer, pb["pool"])
    cand = _read_candidates(pb["candidates_file"])
    P = _prompts(pb["pool"]["n_prompts_train"], pb["pool"]["n_prompts_test"])
    base, _ = _base_by_prompt(REPO_ROOT / pb["pool_dir"])
    m = pb["pool"]["m_swing"]
    torch.manual_seed(pb["optim"]["seed"])
    OUT.mkdir(parents=True, exist_ok=True)

    M = np.full((len(P["train"]), len(cand)), np.nan)
    for pi, prompt in enumerate(P["train"]):
        b = base.get(("train", pi))
        if b is None:
            continue
        for j, instr in enumerate(cand):
            sc = [c for c in generate_batch(model, tok, [prompt] * m, gcfg, system=instr) if c.strip()]
            if sc:
                M[pi, j] = float(np.mean([rm_score(rm, rm_tok, prompt, c) for c in sc])) - b
        if (pi + 1) % 10 == 0:
            print(f"  swing: {pi + 1}/{len(P['train'])} prompts")
    np.savez(OUT / "swing_train.npz", M=M, candidates=np.array(cand, dtype=object))
    print(f"swing matrix -> {OUT / 'swing_train.npz'}  (mean swing {np.nanmean(M):+.3f}, "
          f"best-move mean {np.nanmax(M, axis=1).mean():+.3f})")


# --------------------------------------------------------------- phase: select ----
def _greedy_submodular(M, K):
    """Greedy max of f(S)=Σ_x max(0, max_{p∈S} M[x,p]). Returns (order, f_curve)."""
    n, m = M.shape
    Mc = np.nan_to_num(M, nan=-1e9)
    covered = np.zeros(n)                    # best swing so far per prompt, floored at 0
    order, curve, chosen = [], [], set()
    for _ in range(min(K, m)):
        gains = [(np.maximum(covered, np.maximum(Mc[:, j], 0.0)).sum() if j not in chosen else -1e18)
                 for j in range(m)]
        j = int(np.argmax(gains))
        chosen.add(j); order.append(j)
        covered = np.maximum(covered, np.maximum(Mc[:, j], 0.0))
        curve.append(float(covered.mean()))   # per-prompt mean captured swing at this K
    return order, curve


def phase_select(base_cfg, pb, device, model, tok):
    d = np.load(OUT / "swing_train.npz", allow_pickle=True)
    M, cand = d["M"], list(d["candidates"])
    order, curve = _greedy_submodular(M, pb["select"]["K"])
    sel = [{"rank": i + 1, "cand_idx": int(j), "move": cand[j],
            "mean_captured_swing": curve[i],
            "marginal": curve[i] - (curve[i - 1] if i else 0.0)} for i, j in enumerate(order)]
    json.dump({"order": order, "curve": curve, "selected": sel,
               "unconditional_best_idx": int(np.nanmean(M, axis=0).argmax())},
              open(OUT / "selection.json", "w"), indent=2)
    print("Greedy submodular basis (value-vs-K):")
    for s in sel:
        print(f"  K={s['rank']}: +{s['marginal']:.3f} -> {s['mean_captured_swing']:+.3f}  | {s['move'][:70]}")


# ---------------------------------------------------------------- phase: route ----
def _pca_fit(H, k):
    mu = H.mean(0)
    _, _, Vt = np.linalg.svd(H - mu, full_matrices=False)
    return mu, Vt[:k]


def _train_head(Z, y, n_class, mode, pb, device):
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    if mode == "linear":
        net = nn.Linear(Z.shape[1], n_class)
    else:
        net = nn.Sequential(nn.Linear(Z.shape[1], pb["router"]["mlp_hidden"]), nn.ReLU(),
                            nn.Linear(pb["router"]["mlp_hidden"], n_class))
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=pb["router"]["lr"],
                           weight_decay=pb["router"]["weight_decay"])
    for _ in range(pb["router"]["epochs"]):
        opt.zero_grad(); F.cross_entropy(net(Zt), yt).backward(); opt.step()
    return net


def _predict(net, Z, device):
    with torch.no_grad():
        return net(torch.tensor(Z, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()


def phase_route(base_cfg, pb, device, model, tok, rm, rm_tok):
    steer_layer = base_cfg["steer_layer"]
    gcfg = _gcfg(steer_layer, pb["pool"])
    P = _prompts(pb["pool"]["n_prompts_train"], pb["pool"]["n_prompts_test"])
    base, base_texts = _base_by_prompt(REPO_ROOT / pb["pool_dir"])
    dtr = np.load(OUT / "swing_train.npz", allow_pickle=True)
    Mtr, cand = dtr["M"], list(dtr["candidates"])
    sel = json.load(open(OUT / "selection.json"))
    S = sel["order"]                                  # selected candidate indices, greedy order
    layers = pb["router"]["read_layers"]
    m = pb["eval"]["m_test"]
    torch.manual_seed(pb["optim"]["seed"] + 3)

    # ---- router targets on train: best selected move per prompt, class 0 = none (all <=0) ----
    def targets(M, idxs):
        y, ok = [], []
        for pi in range(M.shape[0]):
            row = M[pi, idxs]
            if np.all(np.isnan(row)):
                ok.append(False); y.append(0); continue
            best = np.nanargmax(row)
            y.append(0 if np.nan_to_num(row[best], nan=-1e9) <= 0 else best + 1)
            ok.append(True)
        return np.array(y), np.array(ok)
    ytr, oktr = targets(Mtr, S)

    # ---- test swing over SELECTED moves only (cheap): M_test + generations for the guard ----
    Mte = np.full((len(P["test"]), len(S)), np.nan)
    te_texts = {}
    for pi, prompt in enumerate(P["test"]):
        b = base.get(("test", pi))
        if b is None:
            continue
        for jj, cidx in enumerate(S):
            sc = [c for c in generate_batch(model, tok, [prompt] * m, gcfg, system=cand[cidx]) if c.strip()]
            if sc:
                Mte[pi, jj] = float(np.mean([rm_score(rm, rm_tok, prompt, c) for c in sc])) - b
                te_texts[(pi, jj)] = sc
        if (pi + 1) % 10 == 0:
            print(f"  route/test-swing: {pi + 1}/{len(P['test'])}")
    yte, okte = targets(Mte, list(range(len(S))))

    # ---- read states at all sweep layers (train + test) ----
    Htr = {L: [] for L in layers}; Hte = {L: [] for L in layers}
    for prompt in P["train"]:
        rs = _read_states_multi(model, tok, prompt, layers)
        for L in layers:
            Htr[L].append(rs[L])
    for prompt in P["test"]:
        rs = _read_states_multi(model, tok, prompt, layers)
        for L in layers:
            Hte[L].append(rs[L])
    n_class = len(S) + 1

    # ---- layer sweep x {linear, mlp}: pick by held-out routing accuracy ----
    def realized(pred):
        vals = [0.0 if pred[pi] == 0 else np.nan_to_num(Mte[pi, pred[pi] - 1], nan=0.0)
                for pi in range(len(P["test"]))]
        return np.array(vals)
    sweep, best = [], None
    for L in layers:
        Xtr = np.stack(Htr[L])[oktr]; Xte = np.stack(Hte[L])
        mu, comp = _pca_fit(Xtr, pb["router"]["n_pca"])
        Ztr = (Xtr - mu) @ comp.T; Zte = (Xte - mu) @ comp.T
        for mode in pb["router"]["arms"]:
            net = _train_head(Ztr, ytr[oktr], n_class, mode, pb, device)
            acc_tr = float((_predict(net, Ztr, device) == ytr[oktr]).mean())
            pred_te = _predict(net, Zte, device)
            acc_te = float((pred_te == yte).mean())
            drm = realized(pred_te)
            rec = {"layer": L, "arm": mode, "acc_tr": acc_tr, "acc_te": acc_te,
                   "drm": float(drm.mean()), "pred": pred_te}
            sweep.append(rec)
            if best is None or rec["acc_te"] > best["acc_te"]:
                best = rec

    _write_report(base_cfg, pb, cand, sel, S, Mtr, Mte, yte, sweep, best, base_texts, te_texts, P)


def _write_report(base_cfg, pb, cand, sel, S, Mtr, Mte, yte, sweep, best, base_texts, te_texts, P):
    mname = base_cfg["base_model"].split("/")[-1]
    # baselines on test
    single = np.nan_to_num(Mte[:, 0], nan=0.0)                       # k=1 greedy move, unconditional
    oracle = np.array([max(0.0, np.nan_to_num(Mte[pi], nan=-1e9).max()) for pi in range(Mte.shape[0])])
    slo, shi = _boot(single); olo, ohi = _boot(oracle)
    bpred = best["pred"]
    drm = np.array([0.0 if bpred[pi] == 0 else np.nan_to_num(Mte[pi, bpred[pi] - 1], nan=0.0)
                    for pi in range(Mte.shape[0])])
    rlo, rhi = _boot(drm)
    frac_none = float((bpred == 0).mean())
    guard = _distinct2([c for v in te_texts.values() for c in v])
    base_d2 = _distinct2([c for pi in range(len(P["test"])) for c in base_texts.get(("test", pi), [])])

    L = [f"# S1 (revised) — procedural-prompt basis + learned router\n",
         f"{mname}. X = {len(cand)} curated candidate moves (subproc.1 abstracted). Swing(x,p)=RM(sys=p)−RM(base), "
         f"base reused from {pb['pool_dir']}. Greedy submodular basis (subproc.2), K={pb['select']['K']}; "
         f"router read-layer swept {pb['router']['read_layers']}, PCA-{pb['router']['n_pca']} (subproc.3). "
         f"n_train={Mtr.shape[0]}, n_test={Mte.shape[0]}, m={pb['pool']['m_swing']}. RM1="
         f"{base_cfg['reward_model'].split('/')[-1]}.\n",
         f"References (A2/7B): prompting +1.08, best-of-n +1.40, contrastive-steering ~0. "
         f"Base distinct-2 {base_d2:.3f}.\n",
         "## Subprocedure 2 — greedy submodular basis (value-vs-K, oracle per-prompt captured swing)",
         "| K | marginal | mean captured swing | move |", "|---|---|---|---|"]
    for s in sel["selected"]:
        L.append(f"| {s['rank']} | +{s['marginal']:.3f} | {s['mean_captured_swing']:+.3f} | {s['move'][:80]} |")
    L += ["", "## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM)",
          "| layer | arm | acc train | acc test | ΔRM1 (realized) |", "|---|---|---|---|---|"]
    for r in sweep:
        star = "  ⟵ best" if r is best else ""
        L.append(f"| {r['layer']} | {r['arm']} | {r['acc_tr']:.2f} | {r['acc_te']:.2f} | {r['drm']:+.3f}{star} |")
    L += ["", "## Held-out result (best router: "
          f"layer {best['layer']}, {best['arm']})",
          f"- **Learned router ΔRM1: {drm.mean():+.3f} [{rlo:+.3f}, {rhi:+.3f}]** (routes to 'none' on "
          f"{frac_none:.0%} of prompts).",
          f"- Best SINGLE unconditional move (k=1): {single.mean():+.3f} [{slo:+.3f}, {shi:+.3f}]  "
          f"— the load-bearing baseline; router must beat this for conditioning to be worth it.",
          f"- Oracle-over-basis ceiling: {oracle.mean():+.3f} [{olo:+.3f}, {ohi:+.3f}].",
          f"- Router-routed generations distinct-2 {guard:.3f} (base {base_d2:.3f}); guard for collapse.",
          "", "## Reading",
          "- **router > best-single-move (CIs)** ⇒ conditioning pays — the prompt-basis method works "
          "where steering's did not. Compare the gap to the oracle-over-basis ceiling (routing quality).",
          "- **router ≈ best-single-move** ⇒ no conditioning value yet: either the routing signal is weak "
          "or X lacks type-specific moves (⇒ build the LLM-from-preferences generator for a stronger X).",
          "- **large train→test accuracy gap** ⇒ router is data-limited at n_train — scale prompts (only "
          "grows the parallel swing precompute, no redesign).",
          "- Judge magnitudes against prompting (+1.08) and best-of-n (+1.40)."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / REPORT).write_text("\n".join(L) + "\n")
    print("\n".join(x for x in L if not x.startswith("|")))
    print(f"\nreport -> {BASIS / REPORT}")


def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["swing", "select", "route", "all"], default="all")
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-config", default=None)
    args = ap.parse_args()
    base_cfg = load_config(args.base_config)
    pb = load_pb_config(args.config)
    tag = pb.get("tag", "")
    if tag:
        OUT = REPO_ROOT / "results" / f"prompt_basis_{tag}"
        REPORT = f"s1_pbasis_{tag}_report.md"
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    rm = rm_tok = None
    if args.phase in ("swing", "route", "all"):
        rm, rm_tok = load_rm(base_cfg, device)
    if args.phase in ("swing", "all"):
        phase_swing(base_cfg, pb, device, model, tok, rm, rm_tok)
    if args.phase in ("select", "all"):
        phase_select(base_cfg, pb, device, model, tok)
    if args.phase in ("route", "all"):
        phase_route(base_cfg, pb, device, model, tok, rm, rm_tok)
    print(log_cost("S1", f"prompt_basis_{args.phase}", time.time() - t0, device,
                   notes="procedural-prompt basis + router (no LLM backprop)"))


if __name__ == "__main__":
    main()
