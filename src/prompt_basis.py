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


def _truncated(t):
    """Proxy for hitting max_new_tokens: completion ends without terminal punctuation."""
    t = t.rstrip()
    return not t.endswith((".", "!", "?", '"', ")", "]", "}", "`", ":"))


# ----------------------------------------------------------------- phase: pool ----
def phase_pool(base_cfg, pb, device, model, tok, rm, rm_tok):
    """Generate a base pool at THIS run's max_new_tokens (detruncated) + RM-score it; logs the
    truncation rate so we can confirm the fix. Shared by the router (base(x)) and by Job B (steering)."""
    steer_layer = base_cfg["steer_layer"]
    gcfg = _gcfg(steer_layer, pb["pool"])
    P = _prompts(pb["pool"]["n_prompts_train"], pb["pool"]["n_prompts_test"])
    mb = pb["pool"]["m_base"]
    torch.manual_seed(pb["optim"]["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    trunc = []
    with open(OUT / "pool.jsonl", "w") as f:
        for split in ("train", "test"):
            for pi, prompt in enumerate(P[split]):
                for c in generate_batch(model, tok, [prompt] * mb, gcfg):
                    if c.strip():
                        trunc.append(_truncated(c))
                        f.write(json.dumps({"split": split, "pi": pi, "prompt": prompt,
                                            "completion": c, "rm": rm_score(rm, rm_tok, prompt, c)}) + "\n")
                f.flush()
    print(f"detrunc base pool -> {OUT / 'pool.jsonl'}  (truncation {np.mean(trunc):.0%} at "
          f"max_new_tokens={pb['pool']['max_new_tokens']}; was 72% at 128)")


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


def _targets(M, idxs):
    """Router class per prompt over columns idxs: 0 = none (best <= 0), else best+1."""
    y, ok = [], []
    for pi in range(M.shape[0]):
        row = M[pi, idxs]
        if np.all(np.isnan(row)):
            ok.append(False); y.append(0); continue
        best = int(np.nanargmax(row))
        y.append(0 if np.nan_to_num(row[best], nan=-1e9) <= 0 else best + 1)
        ok.append(True)
    return np.array(y), np.array(ok)


def phase_route(base_cfg, pb, device, model, tok, rm, rm_tok):
    steer_layer = base_cfg["steer_layer"]
    gcfg = _gcfg(steer_layer, pb["pool"])
    P = _prompts(pb["pool"]["n_prompts_train"], pb["pool"]["n_prompts_test"])
    base, base_texts = _base_by_prompt(REPO_ROOT / pb["pool_dir"])
    dtr = np.load(OUT / "swing_train.npz", allow_pickle=True)
    Mtr, cand = dtr["M"], list(dtr["candidates"])
    sel = json.load(open(OUT / "selection.json"))
    S = sel["order"]
    layers = pb["router"]["read_layers"]
    mt = pb["eval"]["m_test"]
    torch.manual_seed(pb["optim"]["seed"] + 3)
    ytr, oktr = _targets(Mtr, S)                       # router targets (train, full m_swing)

    # ---- test swings over selected moves, SPLIT val/test halves to de-bias the oracle ----
    Mv = np.full((len(P["test"]), len(S)), np.nan); Mt = np.full_like(Mv, np.nan)
    te_texts, trunc = {}, []
    for pi, prompt in enumerate(P["test"]):
        b = base.get(("test", pi))
        if b is None:
            continue
        for jj, cidx in enumerate(S):
            sc = [c for c in generate_batch(model, tok, [prompt] * mt, gcfg, system=cand[cidx]) if c.strip()]
            if len(sc) >= 2:
                r = [rm_score(rm, rm_tok, prompt, c) for c in sc]
                k = max(1, min(mt // 2, len(r) - 1))
                Mv[pi, jj] = float(np.mean(r[:k])) - b; Mt[pi, jj] = float(np.mean(r[k:])) - b
                te_texts[(pi, jj)] = sc; trunc += [_truncated(c) for c in sc]
        if (pi + 1) % 10 == 0:
            print(f"  route/test-swing: {pi + 1}/{len(P['test'])}")
    yte, _ = _targets(Mv, list(range(len(S))))          # test targets from the VAL half

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

    def realized(pred):                                 # score the router's pick on the TEST half
        return np.array([0.0 if pred[pi] == 0 else np.nan_to_num(Mt[pi, pred[pi] - 1], nan=0.0)
                         for pi in range(len(P["test"]))])
    sweep, best = [], None
    for L in layers:
        Xtr = np.stack(Htr[L])[oktr]; Xte = np.stack(Hte[L])
        mu, comp = _pca_fit(Xtr, pb["router"]["n_pca"])
        Ztr = (Xtr - mu) @ comp.T; Zte = (Xte - mu) @ comp.T
        for mode in pb["router"]["arms"]:
            net = _train_head(Ztr, ytr[oktr], n_class, mode, pb, device)
            acc_tr = float((_predict(net, Ztr, device) == ytr[oktr]).mean())
            pred_te = _predict(net, Zte, device)
            rec = {"layer": L, "arm": mode, "acc_tr": acc_tr,
                   "acc_te": float((pred_te == yte).mean()),
                   "drm": float(realized(pred_te).mean()), "pred": pred_te}
            sweep.append(rec)
            if best is None or rec["acc_te"] > best["acc_te"]:
                best = rec
    tr_rate = float(np.mean(trunc)) if trunc else float("nan")
    _write_report(base_cfg, pb, cand, sel, S, Mtr, Mv, Mt, sweep, best, base_texts, te_texts, P, tr_rate)


def _write_report(base_cfg, pb, cand, sel, S, Mtr, Mv, Mt, sweep, best, base_texts, te_texts, P, tr_rate):
    mname = base_cfg["base_model"].split("/")[-1]
    n = Mt.shape[0]
    single = np.nan_to_num(Mt[:, 0], nan=0.0)                          # greedy k=1 move, scored on test half
    naive = np.array([max(0.0, np.nan_to_num(Mt[pi], nan=-1e9).max()) for pi in range(n)])  # biased (same-data max)
    db = []                                                            # DE-BIASED: pick on val, score on test
    for pi in range(n):
        rv = Mv[pi]
        if np.all(np.isnan(rv)):
            continue
        j = int(np.nanargmax(rv)); db.append(max(0.0, np.nan_to_num(Mt[pi, j], nan=0.0)))
    db = np.array(db)
    bpred = best["pred"]
    router = np.array([0.0 if bpred[pi] == 0 else np.nan_to_num(Mt[pi, bpred[pi] - 1], nan=0.0) for pi in range(n)])
    slo, shi = _boot(single); dlo, dhi = _boot(db); rlo, rhi = _boot(router); nlo, nhi = _boot(naive)
    guard = _distinct2([c for v in te_texts.values() for c in v])
    base_d2 = _distinct2([c for pi in range(len(P["test"])) for c in base_texts.get(("test", pi), [])])

    L = ["# S1 (revised) — procedural-prompt basis + router (DETRUNCATED, de-biased)\n",
         f"{mname}. X = {len(cand)} curated moves. Swing(x,p)=RM(sys=p)−RM(base), base regenerated at "
         f"max_new_tokens={pb['pool']['max_new_tokens']} (truncation now {tr_rate:.0%} on move gens; was 72% at 128). "
         f"Greedy submodular basis K={pb['select']['K']}; router read-layer swept {pb['router']['read_layers']}, "
         f"PCA-{pb['router']['n_pca']}. n_train={Mtr.shape[0]}, n_test={n}, m_swing={pb['pool']['m_swing']}, "
         f"m_test={pb['eval']['m_test']} (val/test split). RM1={base_cfg['reward_model'].split('/')[-1]}.\n",
         f"References (A2/7B, TRUNCATED — now suspect): prompting +1.08, best-of-n +1.40, steering ~0. "
         f"Base distinct-2 {base_d2:.3f}.\n",
         "## Subprocedure 2 — greedy submodular basis (value-vs-K; does 'concise' still dominate post-detrunc?)",
         "| K | marginal | mean captured swing | move |", "|---|---|---|---|"]
    for s in sel["selected"]:
        L.append(f"| {s['rank']} | +{s['marginal']:.3f} | {s['mean_captured_swing']:+.3f} | {s['move'][:80]} |")
    L += ["", "## Subprocedure 3 — router layer sweep (held-out routing accuracy & realized ΔRM, test half)",
          "| layer | arm | acc train | acc test | ΔRM1 |", "|---|---|---|---|---|"]
    for r in sweep:
        star = "  ⟵ best" if r is best else ""
        L.append(f"| {r['layer']} | {r['arm']} | {r['acc_tr']:.2f} | {r['acc_te']:.2f} | {r['drm']:+.3f}{star} |")
    L += ["", f"## Held-out result (best router: layer {best['layer']}, {best['arm']}; all scored on the test half)",
          f"- **Learned router ΔRM1: {router.mean():+.3f} [{rlo:+.3f}, {rhi:+.3f}]**.",
          f"- Best SINGLE unconditional move (k=1): {single.mean():+.3f} [{slo:+.3f}, {shi:+.3f}]  "
          f"— the load-bearing baseline; router must beat this for conditioning to pay.",
          f"- **De-biased oracle-over-basis (pick on val, score on test): {db.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]** "
          f"— the TRUE routing ceiling; its gap over the single move is the real conditioning headroom.",
          f"- Naive (biased) oracle for reference: {naive.mean():+.3f} [{nlo:+.3f}, {nhi:+.3f}] "
          f"— inflation vs de-biased = the winner's-curse we removed.",
          f"- Routed generations distinct-2 {guard:.3f} (base {base_d2:.3f}).",
          "", "## Reading",
          "- **truncation now low** confirms the fix; compare the surviving swings to the 128-token run to see "
          "how much single-turn 'headroom' was truncation-avoidance.",
          "- **de-biased oracle ≫ best-single-move** ⇒ real single-turn conditioning headroom exists (then judge "
          "whether the router captures it: router vs single). **de-biased ≈ single** ⇒ conditioning is mostly "
          "illusory here ⇒ carry routing to multi-turn (Subproject 2).",
          "- **router > single (CIs)** ⇒ conditioning pays and is learnable now; **router ≈ single** with a large "
          "de-biased gap ⇒ signal exists but router is data-limited (scale prompts).",
          "- Judge magnitudes against the detruncated single move; the old +1.08/+1.40 refs are truncation-suspect."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / REPORT).write_text("\n".join(L) + "\n")
    print("\n".join(x for x in L if not x.startswith("|")))
    print(f"\nreport -> {BASIS / REPORT}")


def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pool", "swing", "select", "route", "all"], default="all")
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
    if args.phase in ("pool", "swing", "route", "all"):
        rm, rm_tok = load_rm(base_cfg, device)
    if args.phase in ("pool", "all"):
        phase_pool(base_cfg, pb, device, model, tok, rm, rm_tok)
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
