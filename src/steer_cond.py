"""Subproject 1 / S1.2 — conditional controller over a jointly-learned low-rank basis.

delta(x) = a_theta(h(x))^T V, with h(x) = last-token pre-injection residual state at the
read layer (a fixed feature of the frozen base), V in R^{r x d} unit-row + orthogonality +
L1, and a controller a_theta in {global, linear, mlp}. Trained by the same RWR-toward-the-
tilt objective as S1.1 (src/steer_learn.py), reusing its differentiable teacher-forced
injection. Interpretability lives in V (nameable directions + routing map), NOT in the
controller — linear-vs-MLP is a capacity comparison, so both run in one job.

Positive control (type-dependent, known answer): the latent type is a RECOVERABLE linear
feature of the read state (sign of its top-PC projection), and the two types want opposite
poles of a reachable lever (A: hedge+, B: hedge-). Separability is ~100% by construction, so
a global vector must compromise and lose while a conditional r=2 controller that routes by
type should win — and any routing failure is purely optimization, not a signal confound.

    python src/steer_cond.py --phase all [--base-config configs/base_7b.yaml --config ...]

Phases: pool (GPU generate + phi) -> learn (assign types from h, train per arm) -> eval.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models import REPO_ROOT, generate_batch, load_base, load_config, log_cost, resolve_device
from steer_learn import PHI_KEYS, _boot, phi_features, tf_sum_logprob
from steer_sanity import measure_ref_norm

OUT = REPO_ROOT / "results" / "steer_cond"
BASIS = REPO_ROOT / "basis"
REPORT = "s1_cond_report.md"


def load_cond_config(path=None):
    with open(path or (REPO_ROOT / "configs" / "steer_cond.yaml")) as f:
        return yaml.safe_load(f)


def _prompts(n_train, n_test, split="train"):
    raw = json.loads(open(REPO_ROOT / "data" / "prompts.json").read())[split]
    if len(raw) < n_train + n_test:
        raise RuntimeError(f"need {n_train + n_test} prompts, have {len(raw)} in split '{split}'")
    return {"train": raw[:n_train], "test": raw[n_train:n_train + n_test]}


@torch.no_grad()
def read_state(model, tok, prompt, layer):
    """h(x): last-token pre-injection residual state at `layer` (fp32, cpu)."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    cap = {}

    def hook(_m, _i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        model(enc["input_ids"])
    finally:
        h.remove()
    return cap["h"][0, -1, :].float().cpu()


def _assign_types_from_h(model, tok, P, read_layer, device):
    """Define the latent type as a RECOVERABLE linear feature of the read state itself: the sign of
    the projection onto the top principal direction of the train read-states (median split for
    balance). Separability is then ~100% by construction, so a routing failure is purely
    optimization — no artificial cue, no washed-out-signal confound. Returns (types, Htr, Hte)."""
    Htr = torch.stack([read_state(model, tok, p, read_layer) for p in P["train"]])
    Hte = torch.stack([read_state(model, tok, p, read_layer) for p in P["test"]])
    mu = Htr.mean(0)
    _, _, Vh = torch.linalg.svd(Htr - mu, full_matrices=False)
    pc = Vh[0]
    thr = ((Htr - mu) @ pc).median()
    lab = lambda H: ["A" if float((row - mu) @ pc) >= thr else "B" for row in H]
    return {"train": lab(Htr), "test": lab(Hte)}, Htr, Hte


def _type_probe(P, types, Htr, Hte, device):
    """Held-out accuracy of a linear probe read-state -> type (sanity; ~100% by construction here)."""
    def feats(H, split):
        y = torch.tensor([0.0 if t == "A" else 1.0 for t in types[split]])
        return H.to(device), y.to(device)

    Xtr, ytr = feats(Htr, "train"); Xte, yte = feats(Hte, "test")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-3 * w.pow(2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = lambda X, y: float((((X @ w + b) > 0).float() == y).float().mean())
        return acc(Xtr, ytr), acc(Xte, yte)


class Policy(nn.Module):
    """Injection policy: V (r unit directions) + a controller producing coefficients a."""

    def __init__(self, mode, r, d, cap, mlp_hidden=128):
        super().__init__()
        self.mode = mode
        init = 0.5 * cap / (r ** 0.5)       # init ||injection|| ~ 0.5*cap
        V = torch.randn(r, d)
        self.V = nn.Parameter(V / V.norm(dim=1, keepdim=True))
        if mode == "global":
            self.c = nn.Parameter(torch.randn(r) * init)
        elif mode == "linear":
            self.W = nn.Parameter(torch.randn(r, d) * 0.01)
            self.b = nn.Parameter(torch.randn(r) * init)
        elif mode == "mlp":
            self.W1 = nn.Parameter(torch.randn(mlp_hidden, d) * 0.01)
            self.b1 = nn.Parameter(torch.zeros(mlp_hidden))
            self.W2 = nn.Parameter(torch.randn(r, mlp_hidden) * 0.01)
            self.b2 = nn.Parameter(torch.randn(r) * init)
        else:
            raise ValueError(mode)

    def _raw(self, h):
        if self.mode == "global":
            return self.c
        if self.mode == "linear":
            return self.W @ h + self.b
        return self.W2 @ F.relu(self.W1 @ h + self.b1) + self.b2

    def coeff(self, h):
        # UNBOUNDED mixing weights: a<->delta is a bijection on span(V) (no gauge, no tanh
        # saturation), so the routing gradient always flows. Magnitude is shaped by the
        # relu(||delta||-cap)^2 penalty in training + weight decay, not by a squashing function.
        return self._raw(h)

    def delta(self, h):
        return self.coeff(h) @ self.V

    def normalize_(self):
        with torch.no_grad():
            self.V.div_(self.V.norm(dim=1, keepdim=True) + 1e-8)


# ---------------------------------------------------------------- phase: pool ----
def phase_pool(base_cfg, ccfg, device, model, tok):
    layer = base_cfg["steer_layer"]
    pcfg = ccfg["pool"]
    gcfg = {"steer_layer": layer, "generation": {
        "max_new_tokens": pcfg["max_new_tokens"], "do_sample": True,
        "temperature": pcfg["temperature"], "top_p": pcfg["top_p"]}}
    torch.manual_seed(ccfg["optim"]["seed"])
    P = _prompts(pcfg["n_prompts_train"], pcfg["n_prompts_test"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pool.jsonl", "w") as f:
        for split in ("train", "test"):
            for pi, prompt in enumerate(P[split]):
                for c in generate_batch(model, tok, [prompt] * pcfg["n_samples"], gcfg):
                    if c.strip():
                        f.write(json.dumps({"split": split, "pi": pi, "prompt": prompt,
                                            "completion": c, "phi": phi_features(c)}) + "\n")
                f.flush()
    print(f"pool -> {OUT / 'pool.jsonl'}")


def _load_pool():
    rows = [json.loads(l) for l in open(OUT / "pool.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["pi"]), []).append(r)
    return rows, by


def _typed_reward(rows, ccfg):
    """z-score phi over the TRAIN pool; R(phi,type) = <e*_type, phi_std>."""
    E = {t: np.array(v, float) for t, v in ccfg["reward"]["types"].items()}
    tr = np.array([[r["phi"][k] for k in PHI_KEYS] for r in rows if r["split"] == "train"], float)
    mu, sd = tr.mean(0), tr.std(0) + 1e-8

    def R(phi, typ):
        v = np.array([phi[k] for k in PHI_KEYS], float)
        return float(E[typ] @ ((v - mu) / sd))
    return R, {"mean": mu.tolist(), "std": sd.tolist(), "types": ccfg["reward"]["types"]}


# --------------------------------------------------------------- phase: learn ----
def _train_arm(mode, cells, d, cap, ccfg, layer):
    p = ccfg["policy"]
    torch.manual_seed(ccfg["optim"]["seed"])
    pol = Policy(mode, p["rank"], d, cap, p["mlp_hidden"]).to(cells[0][0].device)
    opt = torch.optim.Adam(pol.parameters(), lr=ccfg["optim"]["lr"],
                           weight_decay=ccfg["optim"].get("weight_decay", 0.0))
    l1, orth, mag_pen = p["l1"], p["orth"], p.get("mag_penalty", 0.1)
    for epoch in range(ccfg["optim"]["epochs"]):
        tot = 0.0
        for i in torch.randperm(len(cells)).tolist():
            h, prefix, comp_ids, w = cells[i]
            delta = pol.delta(h)
            lp = tf_sum_logprob(_MODEL, _TOK, prefix, comp_ids, layer, delta)
            # soft magnitude cap (relu penalty above cap) shapes ||delta||; no gauge, no saturation
            loss = -(w * lp).sum() + mag_pen * torch.relu(delta.norm() - cap) ** 2 + l1 * pol.V.abs().sum()
            if orth and p["rank"] > 1:
                G = pol.V @ pol.V.t()
                loss = loss + orth * (G - torch.diag(torch.diag(G))).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            pol.normalize_()
            tot += float(loss.item())
        print(f"  [{mode}] epoch {epoch}: mean loss {tot / len(cells):.3f}")
    return pol


def phase_learn(base_cfg, ccfg, device, model, tok):
    global _MODEL, _TOK
    _MODEL, _TOK = model, tok
    layer = base_cfg["steer_layer"]
    read_layer = ccfg["policy"]["read_layer"]
    rows, by = _load_pool()
    R, stats = _typed_reward(rows, ccfg)
    P = _prompts(ccfg["pool"]["n_prompts_train"], ccfg["pool"]["n_prompts_test"])
    d = model.config.hidden_size
    ref = measure_ref_norm(model, tok, P["train"][:16], layer)
    cap = ccfg["policy"]["mag_cap_frac"] * ref
    print(f"ref_norm(layer {layer}) = {ref:.1f}; mag cap = {cap:.1f}")

    types, Htr, Hte = _assign_types_from_h(model, tok, P, read_layer, device)
    tr_acc, te_acc = _type_probe(P, types, Htr, Hte, device)
    print(f"type-separability probe: train {tr_acc:.0%}, test {te_acc:.0%}")
    model.requires_grad_(False)

    cells = []
    for pi, prompt in enumerate(P["train"]):
        pool = [x for x in by.get(("train", pi), []) if x["completion"].strip()]
        if len(pool) < 2:
            continue
        prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True)["input_ids"]
        comp_ids = [tok(x["completion"], return_tensors="pt",
                        add_special_tokens=False)["input_ids"] for x in pool]
        Rk = torch.tensor([R(x["phi"], types["train"][pi]) for x in pool])
        w = torch.softmax(Rk / ccfg["reward"]["beta"], dim=0).to(device)
        cells.append((Htr[pi].to(device), prefix, comp_ids, w))

    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ccfg["policy"]["arms"]:
        pol = _train_arm(mode, cells, d, cap, ccfg, layer)
        torch.save(pol.state_dict(), OUT / f"policy_{mode}.pt")
    json.dump({**stats, "ref": ref, "cap": cap, "d": d, "type_labels": types,
               "probe": [tr_acc, te_acc]}, open(OUT / "cond_stats.json", "w"))
    print(f"policies -> {OUT}/policy_*.pt")


# ---------------------------------------------------------------- phase: eval ----
def phase_eval(base_cfg, ccfg, device, model, tok):
    layer = base_cfg["steer_layer"]
    read_layer = ccfg["policy"]["read_layer"]
    rows, by = _load_pool()
    st = json.load(open(OUT / "cond_stats.json"))
    mu, sd = np.array(st["mean"]), np.array(st["std"])
    E = {t: np.array(v, float) for t, v in st["types"].items()}
    d, cap = st["d"], st["cap"]
    types, te_acc = st["type_labels"], st["probe"][1]

    def R(phi, typ):
        v = np.array([phi[k] for k in PHI_KEYS], float)
        return float(E[typ] @ ((v - mu) / sd))

    P = _prompts(ccfg["pool"]["n_prompts_train"], ccfg["pool"]["n_prompts_test"])
    gcfg = {"steer_layer": layer, "generation": {
        "max_new_tokens": ccfg["pool"]["max_new_tokens"], "do_sample": True,
        "temperature": ccfg["pool"]["temperature"], "top_p": ccfg["pool"]["top_p"]}}
    torch.manual_seed(ccfg["optim"]["seed"] + 7)

    # precompute test read-states + base reward/phi per prompt
    tstate, base_R, base_phi, typ_of = {}, {}, {}, {}
    for pi, prompt in enumerate(P["test"]):
        bpool = [x for x in by.get(("test", pi), []) if x["completion"].strip()]
        if not bpool:
            continue
        typ = types["test"][pi]
        tstate[pi] = read_state(model, tok, prompt, read_layer).to(device)
        base_R[pi] = np.mean([R(x["phi"], typ) for x in bpool])
        base_phi[pi] = np.mean([[x["phi"][k] for k in PHI_KEYS] for x in bpool], axis=0)
        typ_of[pi] = typ

    mname = base_cfg["base_model"].split("/")[-1]
    lines = ["# S1.2 — conditional steering controller (type-dependent positive control)\n",
             f"Types = sign of the read state's top-PC projection (recoverable by construction); "
             f"A→hedge+, B→hedge−. {mname}, steer L{layer}, read L{read_layer}, rank {ccfg['policy']['rank']}, "
             f"soft mag cap {cap:.0f} (unbounded coeffs, penalty-shaped). n={ccfg['pool']['n_samples']} pool. "
             "Δ-R = on-policy steered − base, held-out.\n",
             f"**Type-separability probe** (linear h(x)→type): held-out **{te_acc:.0%}** "
             "(~100% by construction ⇒ a routing failure is purely optimization).\n",
             "| arm | Δ-R [95% CI] | Δ-R type A | Δ-R type B |",
             "|---|---|---|---|"]
    arm_results = {}
    for mode in ccfg["policy"]["arms"]:
        pol = Policy(mode, ccfg["policy"]["rank"], d, cap, ccfg["policy"]["mlp_hidden"]).to(device)
        pol.load_state_dict(torch.load(OUT / f"policy_{mode}.pt")); pol.eval()
        dR, dR_t = [], {"A": [], "B": []}
        coeff_t = {"A": [], "B": []}
        sphi_t = {"A": [], "B": []}
        for pi in tstate:
            with torch.no_grad():
                delta = pol.delta(tstate[pi])
                dn = delta.norm()
                if dn > cap:                       # safety clamp at eval (training holds ||delta||~cap)
                    delta = delta * (cap / dn)
                coeff_t[typ_of[pi]].append(pol.coeff(tstate[pi]).cpu().numpy())
            scomp = generate_batch(model, tok, [P["test"][pi]] * ccfg["pool"]["n_samples"],
                                   gcfg, vector=delta, alpha=1.0)
            sphi = [phi_features(c) for c in scomp if c.strip()]
            sR = np.mean([R(ph, typ_of[pi]) for ph in sphi])
            dR.append(sR - base_R[pi]); dR_t[typ_of[pi]].append(sR - base_R[pi])
            sphi_t[typ_of[pi]].append(np.mean([[ph[k] for k in PHI_KEYS] for ph in sphi], axis=0))
        dR = np.array(dR)
        lo, hi = _boot(dR)
        lines.append(f"| {mode} | {dR.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{np.mean(dR_t['A']):+.3f} | {np.mean(dR_t['B']):+.3f} |")
        V_np = pol.V.detach().cpu().numpy()
        dbar = {t: np.mean(coeff_t[t], 0) @ V_np for t in "AB"}     # mean injection direction per type
        rc = float(dbar["A"] @ dbar["B"] / (np.linalg.norm(dbar["A"]) * np.linalg.norm(dbar["B"]) + 1e-9))
        arm_results[mode] = {"dR": dR.mean(), "route_cos": rc,
                             "coeff": {t: np.mean(coeff_t[t], 0).tolist() for t in "AB"},
                             "sphi": {t: np.mean(sphi_t[t], 0).tolist() for t in "AB"}}

    # routing: does the controller send different coefficients to A vs B?
    lines.append("\n## Routing — cosine between mean injection direction for type A vs B "
                 "(≈+1 = same direction = no routing; ≤0 = opposite = routing)")
    for mode in ccfg["policy"]["arms"]:
        cA, cB = np.round(arm_results[mode]["coeff"]["A"], 2), np.round(arm_results[mode]["coeff"]["B"], 2)
        lines.append(f"- **{mode}**: cos(δ̄A, δ̄B) = **{arm_results[mode]['route_cos']:+.2f}**  "
                     f"(a|A={cA.tolist()}, a|B={cB.tolist()})")

    # recovery: per type, realized phi shift should favor that type's lever
    lines.append("\n## Recovery — realized φ (steered − base) by type; A wants hedge↑, B wants hedge↓")
    bphi_t = {t: np.mean([base_phi[pi] for pi in tstate if typ_of[pi] == t], 0) for t in "AB"}
    for mode in ccfg["policy"]["arms"]:
        for t in "AB":
            dphi = np.array(arm_results[mode]["sphi"][t]) - bphi_t[t]
            lines.append(f"- **{mode}** type {t}: Δwords {dphi[0]:+.2f}, "
                         f"Δhedge {dphi[1]:+.2f}, Δquestions {dphi[2]:+.2f}")

    g = arm_results["global"]["dR"]
    cond = [m for m in ccfg["policy"]["arms"] if m != "global"]
    best = max(cond, key=lambda m: arm_results[m]["dR"], default="global")
    ROUTE_MAX = 0.5     # cos(δ̄A,δ̄B) below this = the two types get genuinely different directions
    routed = [m for m in cond if arm_results[m]["route_cos"] < ROUTE_MAX and arm_results[m]["dR"] > g]
    green = bool(routed) and te_acc > 0.75
    lines += ["\n## Reading",
              f"Conditioning value = best conditional Δ-R ({arm_results[best]['dR']:+.3f}, {best}) − "
              f"global Δ-R ({g:+.3f}) = **{arm_results[best]['dR'] - g:+.3f}**.",
              "Routing cos(δ̄A,δ̄B): " + ", ".join(f"{m} {arm_results[m]['route_cos']:+.2f}" for m in cond) +
              f" (< {ROUTE_MAX} = genuine per-type routing, not one stronger global vector).",
              f"**S1.2 {'GREEN' if green else 'NOT green'}**: requires a conditional arm routing to "
              f"different directions per type (cos < {ROUTE_MAX}) AND Δ-R > global AND a legible type "
              f"signal (probe > 75%). "
              + (f"Met by: {routed}." if green else
                 "Not met — an arm that beats global without directional routing is just one stronger "
                 "GLOBAL vector (prompt-distribution artifact); a low probe would mean the cue is washed out.")]
    BASIS.mkdir(exist_ok=True)
    (BASIS / REPORT).write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if not l.startswith("|")))
    print(f"\nreport -> {BASIS / REPORT}")


def main():
    global OUT, REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pool", "learn", "eval", "all"], default="all")
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-config", default=None, help="base model config, e.g. configs/base_7b.yaml")
    args = ap.parse_args()
    base_cfg = load_config(args.base_config)
    ccfg = load_cond_config(args.config)
    tag = ccfg.get("tag", "")
    if tag:                                   # variant (e.g. 7b) writes separate artifacts
        OUT = REPO_ROOT / "results" / f"steer_cond_{tag}"
        REPORT = f"s1_cond_{tag}_report.md"
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    if args.phase in ("pool", "all"):
        phase_pool(base_cfg, ccfg, device, model, tok)
    if args.phase in ("learn", "all"):
        phase_learn(base_cfg, ccfg, device, model, tok)
    if args.phase in ("eval", "all"):
        phase_eval(base_cfg, ccfg, device, model, tok)
    print(log_cost("S1", f"steer_cond_{args.phase}", time.time() - t0, device,
                   notes="conditional controller, type-dependent positive control"))


if __name__ == "__main__":
    main()
