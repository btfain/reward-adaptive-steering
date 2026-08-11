"""Subproject 1 / S1.2 (real RM) — learn reward-driven steering against the REAL reward model.

RWR toward the KL-tilt (per-prompt w_k = softmax(RM_k / beta)) trains a steering policy
(global / linear / mlp conditional controller over a jointly-learned low-rank sparse-orthogonal
basis) to maximize the given RM on UltraFeedback prompts, base frozen. Reuses S1.1's differentiable
teacher-forced injection (steer_learn.tf_sum_logprob) and S1.2's Policy (steer_cond.Policy).

The core Subproject-1 question — can LEARNED steering beat the contrastive ~0 headroom (A2) at 7B?
The GLOBAL arm answers it independent of the routing/conditioning question; the conditional arms add
the (routing-dependent) conditioning test. Evaluated on-policy, held-out, paired vs base under BOTH
RMs (variance control), against the best-of-n ceiling.

    python src/steer_rm.py --phase all --base-config configs/base_7b.yaml --config configs/steer_rm_7b.yaml

Phases: pool (generate + RM1-score) -> learn (per arm) -> eval (on-policy ΔRM, both RMs, best-of-n).
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
from steer_sanity import measure_ref_norm

OUT = REPO_ROOT / "results" / "steer_rm"
BASIS = REPO_ROOT / "basis"
REPORT = "s1_rm_report.md"


def load_rm_config(path=None):
    with open(path or (REPO_ROOT / "configs" / "steer_rm.yaml")) as f:
        return yaml.safe_load(f)


def _gcfg(layer, pcfg):
    return {"steer_layer": layer, "generation": {
        "max_new_tokens": pcfg["max_new_tokens"], "do_sample": True,
        "temperature": pcfg["temperature"], "top_p": pcfg["top_p"]}}


def _load_pool():
    rows = [json.loads(l) for l in open(OUT / "pool.jsonl")]
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["pi"]), []).append(r)
    return rows, by


# ---------------------------------------------------------------- phase: pool ----
def phase_pool(base_cfg, rcfg, device, model, tok, rm, rm_tok):
    layer = base_cfg["steer_layer"]
    pcfg = rcfg["pool"]
    gcfg = _gcfg(layer, pcfg)
    torch.manual_seed(rcfg["optim"]["seed"])
    P = _prompts(pcfg["n_prompts_train"], pcfg["n_prompts_test"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pool.jsonl", "w") as f:
        for split in ("train", "test"):
            for pi, prompt in enumerate(P[split]):
                for c in generate_batch(model, tok, [prompt] * pcfg["n_samples"], gcfg):
                    if c.strip():
                        f.write(json.dumps({"split": split, "pi": pi, "prompt": prompt,
                                            "completion": c,
                                            "rm": rm_score(rm, rm_tok, prompt, c)}) + "\n")
                f.flush()
    print(f"pool -> {OUT / 'pool.jsonl'}")


# --------------------------------------------------------------- phase: learn ----
def _train_arm(mode, cells, d, cap, rcfg, layer, model, tok):
    p = rcfg["policy"]
    torch.manual_seed(rcfg["optim"]["seed"])
    pol = Policy(mode, p["rank"], d, cap, p["mlp_hidden"]).to(cells[0][0].device)
    opt = torch.optim.Adam(pol.parameters(), lr=rcfg["optim"]["lr"],
                           weight_decay=rcfg["optim"].get("weight_decay", 0.0))
    l1, orth, mag_pen = p["l1"], p["orth"], p.get("mag_penalty", 0.1)
    for epoch in range(rcfg["optim"]["epochs"]):
        tot = 0.0
        for i in torch.randperm(len(cells)).tolist():
            h, prefix, comp_ids, w = cells[i]
            delta = pol.delta(h)
            lp = tf_sum_logprob(model, tok, prefix, comp_ids, layer, delta)
            loss = -(w * lp).sum() + mag_pen * torch.relu(delta.norm() - cap) ** 2 + l1 * pol.V.abs().sum()
            if orth and p["rank"] > 1:
                G = pol.V @ pol.V.t()
                loss = loss + orth * (G - torch.diag(torch.diag(G))).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            pol.normalize_()
            tot += float(loss.item())
        print(f"  [{mode}] epoch {epoch}: mean loss {tot / len(cells):.3f}")
    return pol


def phase_learn(base_cfg, rcfg, device, model, tok):
    layer = base_cfg["steer_layer"]
    read_layer = rcfg["policy"]["read_layer"]
    rows, by = _load_pool()
    P = _prompts(rcfg["pool"]["n_prompts_train"], rcfg["pool"]["n_prompts_test"])
    d = model.config.hidden_size
    ref = measure_ref_norm(model, tok, P["train"][:16], layer)
    cap = rcfg["policy"]["mag_cap_frac"] * ref
    print(f"ref_norm(layer {layer}) = {ref:.1f}; mag cap = {cap:.1f}")
    model.requires_grad_(False)

    cells = []
    for pi, prompt in enumerate(P["train"]):
        pool = [x for x in by.get(("train", pi), []) if x["completion"].strip()]
        if len(pool) < 2:
            continue
        h = read_state(model, tok, prompt, read_layer).to(device)
        prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True)["input_ids"]
        comp_ids = [tok(x["completion"], return_tensors="pt",
                        add_special_tokens=False)["input_ids"] for x in pool]
        Rk = torch.tensor([x["rm"] for x in pool])
        w = torch.softmax(Rk / rcfg["reward"]["beta"], dim=0).to(device)
        cells.append((h, prefix, comp_ids, w))

    OUT.mkdir(parents=True, exist_ok=True)
    for mode in rcfg["policy"]["arms"]:
        pol = _train_arm(mode, cells, d, cap, rcfg, layer, model, tok)
        torch.save(pol.state_dict(), OUT / f"policy_{mode}.pt")
    json.dump({"ref": ref, "cap": cap, "d": d}, open(OUT / "stats.json", "w"))
    print(f"policies -> {OUT}/policy_*.pt")


# ---------------------------------------------------------------- phase: eval ----
def _cond_diversity(pol, states, device, cap):
    """Mean cosine of each prompt's injection direction to the population-mean direction.
    ~1 => the controller acts ~globally (little conditioning); <1 => it varies by prompt."""
    ds = []
    for h in states.values():
        with torch.no_grad():
            v = pol.delta(h)
            n = v.norm()
            ds.append((v * (cap / n) if n > cap else v).cpu().numpy())
    D = np.array(ds)
    mean = D.mean(0)
    mean /= np.linalg.norm(mean) + 1e-9
    cs = [float(v @ mean / (np.linalg.norm(v) + 1e-9)) for v in D]
    return float(np.mean(cs))


def phase_eval(base_cfg, rcfg, device, model, tok, rm, rm_tok, rm2, rm2_tok):
    layer = base_cfg["steer_layer"]
    read_layer = rcfg["policy"]["read_layer"]
    rows, by = _load_pool()
    st = json.load(open(OUT / "stats.json"))
    d, cap = st["d"], st["cap"]
    P = _prompts(rcfg["pool"]["n_prompts_train"], rcfg["pool"]["n_prompts_test"])
    gcfg = _gcfg(layer, rcfg["pool"])
    m = rcfg["eval"]["n_samples"]
    torch.manual_seed(rcfg["optim"]["seed"] + 7)

    # base RM1/RM2 per held-out prompt (RM1 from pool; RM2 scored here); best-of-n ceiling on test
    tstate, base1, base2, bo = {}, {}, {}, []
    for pi, prompt in enumerate(P["test"]):
        bpool = [x for x in by.get(("test", pi), []) if x["completion"].strip()]
        if not bpool:
            continue
        tstate[pi] = read_state(model, tok, prompt, read_layer).to(device)
        r1 = [x["rm"] for x in bpool]
        base1[pi] = float(np.mean(r1))
        base2[pi] = float(np.mean([rm_score(rm2, rm2_tok, prompt, x["completion"]) for x in bpool]))
        bo.append(max(r1) - np.mean(r1))
    bo_n = float(np.mean(bo))

    mname = base_cfg["base_model"].split("/")[-1]
    lines = ["# S1.2 (real RM) — learned reward-driven steering vs the given reward model\n",
             f"{mname}, steer L{layer}, read L{read_layer}, rank {rcfg['policy']['rank']}, soft mag cap "
             f"{cap:.0f}. RWR (w=softmax(RM/β)) on {rcfg['pool']['n_prompts_train']} prompts, n={rcfg['pool']['n_samples']} "
             f"pool. Δ-RM = on-policy steered − base, held-out, paired, m={m} samples. RM1="
             f"{base_cfg['reward_model'].split('/')[-1]}, RM2={base_cfg['reward_model_alt'].split('/')[-1]}.\n",
             f"Reference points (A2, 7B): contrastive-steering headroom **~0** (+0.15 [−0.04,+0.34]); "
             f"prompting **+1.08**; best-of-{rcfg['pool']['n_samples']} ceiling here **{bo_n:+.2f}** (RM1).\n",
             "| arm | ΔRM1 [95% CI] | ΔRM2 [95% CI] | cond. globalness |",
             "|---|---|---|---|"]
    arm_res = {}
    for mode in rcfg["policy"]["arms"]:
        pol = Policy(mode, rcfg["policy"]["rank"], d, cap, rcfg["policy"]["mlp_hidden"]).to(device)
        pol.load_state_dict(torch.load(OUT / f"policy_{mode}.pt")); pol.eval()
        d1, d2 = [], []
        for pi in tstate:
            with torch.no_grad():
                delta = pol.delta(tstate[pi])
                dn = delta.norm()
                if dn > cap:
                    delta = delta * (cap / dn)
            sc = [c for c in generate_batch(model, tok, [P["test"][pi]] * m, gcfg,
                                            vector=delta, alpha=1.0) if c.strip()]
            d1.append(np.mean([rm_score(rm, rm_tok, P["test"][pi], c) for c in sc]) - base1[pi])
            d2.append(np.mean([rm_score(rm2, rm2_tok, P["test"][pi], c) for c in sc]) - base2[pi])
        d1, d2 = np.array(d1), np.array(d2)
        lo1, hi1 = _boot(d1); lo2, hi2 = _boot(d2)
        glob = _cond_diversity(pol, tstate, device, cap) if mode != "global" else 1.0
        lines.append(f"| {mode} | {d1.mean():+.3f} [{lo1:+.3f}, {hi1:+.3f}] | "
                     f"{d2.mean():+.3f} [{lo2:+.3f}, {hi2:+.3f}] | {glob:.2f} |")
        arm_res[mode] = {"d1": float(d1.mean()), "lo1": lo1, "glob": glob}

    best = max(rcfg["policy"]["arms"], key=lambda a: arm_res[a]["d1"])
    lines += ["\n## Reading",
              f"Best arm: **{best}** at ΔRM1 {arm_res[best]['d1']:+.3f} [{arm_res[best]['lo1']:+.3f}, …]. "
              "The load-bearing result is the **global** arm vs the A2 contrastive ~0: if global's CI "
              "excludes 0, LEARNED steering beats contrastive extraction at 7B (the Subproject-1 claim). "
              "Conditional arms add value only if they beat global AND actually condition (globalness < 1); "
              "their trust depends on the synthetic routing positive control. Judge magnitude against "
              "prompting (+1.08) and the best-of-n ceiling above — steering that captures a fraction of "
              "best-of-n at 1× inference, interpretably, is the win; a clean null bounds steering's ceiling."]
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
    rcfg = load_rm_config(args.config)
    tag = rcfg.get("tag", "")
    if tag:
        OUT = REPO_ROOT / "results" / f"steer_rm_{tag}"
        REPORT = f"s1_rm_{tag}_report.md"
    device = resolve_device(base_cfg)
    t0 = time.time()
    model, tok = load_base(base_cfg, device)
    rm, rm_tok = load_rm(base_cfg, device)
    if args.phase in ("pool", "all"):
        phase_pool(base_cfg, rcfg, device, model, tok, rm, rm_tok)
    if args.phase in ("learn", "all"):
        phase_learn(base_cfg, rcfg, device, model, tok)
    if args.phase in ("eval", "all"):
        alt = {**base_cfg, "reward_model": base_cfg["reward_model_alt"]}
        rm2, rm2_tok = load_rm(alt, device)
        phase_eval(base_cfg, rcfg, device, model, tok, rm, rm_tok, rm2, rm2_tok)
    print(log_cost("S1", f"steer_rm_{args.phase}", time.time() - t0, device, notes="real RM"))


if __name__ == "__main__":
    main()
