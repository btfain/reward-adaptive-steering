"""B1 — online contextual-bandit router at scale: FROZEN-head vs FINE-TUNED-encoder (controlled,
same encoder). The definitive single-turn test: does fine-tuning at 1500 prompts break the ~+0.38
frozen ceiling that B0.5 pinned across encoders/dims/data? Reuses the fixed 8-move basis (no new
selection). Hardened online REINFORCE (norm-adv + value baseline + entropy) validated offline in B0.

Reward for prompt x under move a = RM(generate(x, system=move_a)) - base_ref(x); decline arm = 0.
Only the sampled arm is generated (the bandit's cost win: ~E gens/prompt, not K*m_swing). NO backprop
through the 7B — grad flows only into the router (and, in the fine-tune arm, the small encoder).

Phases (pool + prep are shardable for parallel generation; train runs one job PER ARM; report is I/O):
  pool  --shard i/N   base pool (m_base gens) for ALL prompts -> base_pool_shard_i.jsonl
  assemble            merge base pool -> base_ref.json
  prep  --shard i/N   ceiling K x m_test (de-biased oracle) + single-move eval baseline -> prep_shard_i.npz
  assemble_prep       merge -> oracle.json (de-biased oracle + single baseline, per-prompt)
  train --arm frozen|finetune   online bandit; per-epoch entropy/usage/reward log; router-eval -> train_<arm>.json
  report             frozen vs finetune vs single vs oracle + anti-collapse + cost -> basis report

    python src/bandit_online.py --phase pool --shard 0/4 --base-config configs/base_7b.yaml --config configs/bandit_online_1500_7b.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModel, AutoTokenizer

from models import (REPO_ROOT, generate_batch, load_base, load_config, load_rm,
                    log_cost, resolve_device, rm_score)

BASIS = REPO_ROOT / "basis"


def _pb(path):
    with open(REPO_ROOT / path if not Path(path).is_absolute() else path) as f:
        return yaml.safe_load(f)


def _out(pb):
    d = REPO_ROOT / "results" / f"bandit_online_{pb['tag']}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gcfg(base_cfg, pool):
    return {"steer_layer": base_cfg["steer_layer"],
            "generation": {"max_new_tokens": pool["max_new_tokens"], "do_sample": True,
                           "temperature": pool["temperature"], "top_p": pool["top_p"]}}


def _moves(pb):
    sel = json.load(open(REPO_ROOT / pb["basis_selection"]))
    return [s["move"] for s in sel["selected"]]                     # K moves in selection order


def _all_prompts(pb):
    """All B1 prompts + role boundaries. Global index gi: [0,n_tr) train, [n_tr,n_tr+n_ev) eval, rest ceiling."""
    n_tr, n_ev, n_ce = pb["n_train"], pb["n_router_eval"], pb["n_ceiling"]
    raw = json.load(open(REPO_ROOT / "data" / "prompts.json"))[pb["prompts_split"]]
    need = n_tr + n_ev + n_ce
    if len(raw) < need:
        raise RuntimeError(f"split '{pb['prompts_split']}' has {len(raw)} < {need} prompts")
    return raw[:need], (n_tr, n_ev, n_ce)


def _role(gi, bnds):
    n_tr, n_ev, _ = bnds
    return "train" if gi < n_tr else ("eval" if gi < n_tr + n_ev else "ceiling")


def _keep(gi, shard):
    return shard is None or (gi % shard[1] == shard[0])


def _boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = np.sort(v[rng.integers(0, len(v), (n, len(v)))].mean(1))
    return float(b[int(0.025 * n)]), float(b[int(0.975 * n)])


# --------------------------------------------------------------------- pool ----
def phase_pool(base_cfg, pb, model, tok, rm, rm_tok, shard):
    P, bnds = _all_prompts(pb)
    gcfg = _gcfg(base_cfg, pb["pool"]); mb = pb["pool"]["m_base"]
    torch.manual_seed(pb["optim"]["seed"] + (0 if shard is None else shard[0] + 1))
    out = _out(pb) / (f"base_pool_shard_{shard[0]}.jsonl" if shard else "base_pool.jsonl")
    with open(out, "w") as f:
        for gi, prompt in enumerate(P):
            if not _keep(gi, shard):
                continue
            for c in generate_batch(model, tok, [prompt] * mb, gcfg):
                if c.strip():
                    f.write(json.dumps({"gi": gi, "role": _role(gi, bnds), "prompt": prompt,
                                        "completion": c, "rm": rm_score(rm, rm_tok, prompt, c)}) + "\n")
            f.flush()
    print(f"base pool {'shard '+str(shard[0]) if shard else ''} -> {out}")


def assemble_pool(pb):
    O = _out(pb)
    shards = sorted(O.glob("base_pool_shard_*.jsonl"))
    rows = [l for s in shards for l in open(s)] if shards else list(open(O / "base_pool.jsonl"))
    by = {}
    for l in rows:
        x = json.loads(l)
        by.setdefault(x["gi"], []).append(x["rm"])
    base_ref = {int(gi): float(np.mean(v)) for gi, v in by.items()}
    json.dump(base_ref, open(O / "base_ref.json", "w"))
    print(f"assembled base_ref for {len(base_ref)} prompts -> {O/'base_ref.json'}")


# --------------------------------------------------------------------- prep ----
def phase_prep(base_cfg, pb, model, tok, rm, rm_tok, shard):
    """Ceiling: K x m_test swings per ceiling prompt (de-biased oracle). Single: move0 swing per eval
    prompt (the router's baseline). Both need base_ref (run after assemble)."""
    P, bnds = _all_prompts(pb)
    O = _out(pb); base_ref = {int(k): v for k, v in json.load(open(O / "base_ref.json")).items()}
    moves = _moves(pb); K = len(moves); mt = pb["eval"]["m_test"]
    gcfg = _gcfg(base_cfg, pb["pool"])
    torch.manual_seed(pb["optim"]["seed"] + 100 + (0 if shard is None else shard[0]))
    ceil_gi, ceil_sw = [], []                                       # ceiling: (n, K, m_test) swings
    single_gi, single_sw = [], []                                   # eval: move0 swing (m=1)
    for gi, prompt in enumerate(P):
        role = _role(gi, bnds)
        if not _keep(gi, shard) or base_ref.get(gi) is None:
            continue
        b = base_ref[gi]
        if role == "ceiling":
            sw = np.full((K, mt), np.nan)
            for j, mv in enumerate(moves):
                comps = generate_batch(model, tok, [prompt] * mt, gcfg, system=mv)
                for t, c in enumerate(comps):
                    if c.strip():
                        sw[j, t] = rm_score(rm, rm_tok, prompt, c) - b
            ceil_gi.append(gi); ceil_sw.append(sw)
        elif role == "eval":
            c = generate_batch(model, tok, [prompt], gcfg, system=moves[0])[0]
            if c.strip():
                single_gi.append(gi); single_sw.append(rm_score(rm, rm_tok, prompt, c) - b)
    np.savez(O / (f"prep_shard_{shard[0]}.npz" if shard else "prep.npz"),
             ceil_gi=np.array(ceil_gi), ceil_sw=np.array(ceil_sw) if ceil_sw else np.zeros((0, K, mt)),
             single_gi=np.array(single_gi), single_sw=np.array(single_sw))
    print(f"prep {'shard '+str(shard[0]) if shard else ''}: {len(ceil_gi)} ceiling, {len(single_gi)} single-eval")


def assemble_prep(pb):
    O = _out(pb)
    shards = sorted(O.glob("prep_shard_*.npz")) or [O / "prep.npz"]
    cg, cs, sg, ss = [], [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=True)
        if len(z["ceil_gi"]):
            cg.append(z["ceil_gi"]); cs.append(z["ceil_sw"])
        if len(z["single_gi"]):
            sg.append(z["single_gi"]); ss.append(z["single_sw"])
    ceil_sw = np.concatenate(cs) if cs else np.zeros((0, len(_moves(pb)), pb["eval"]["m_test"]))
    single_sw = np.concatenate(ss) if ss else np.array([])
    # de-biased oracle: val-half selects best move, test-half scores it, decline if <=0
    h = ceil_sw.shape[2] // 2
    oracle = []
    for sw in ceil_sw:                                              # (K, m_test)
        val, test = np.nanmean(sw[:, :h], 1), np.nanmean(sw[:, h:], 1)
        if np.all(np.isnan(val)):
            continue
        j = int(np.nanargmax(np.nan_to_num(val, nan=-1e9)))
        oracle.append(max(0.0, float(np.nan_to_num(test[j], nan=0.0))))
    res = {"oracle_mean": float(np.mean(oracle)) if oracle else float("nan"),
           "oracle_ci": _boot(oracle), "n_ceiling": len(oracle),
           "single_mean": float(np.mean(single_sw)) if len(single_sw) else float("nan"),
           "single_ci": _boot(single_sw), "n_single": int(len(single_sw)),
           "single_by_gi": {int(g): float(v) for g, v in zip(np.concatenate(sg), single_sw)} if sg else {}}
    json.dump(res, open(O / "oracle.json", "w"))
    print(f"oracle (de-biased) {res['oracle_mean']:+.3f}  single {res['single_mean']:+.3f}  "
          f"({res['n_ceiling']} ceiling / {res['n_single']} eval) -> {O/'oracle.json'}")


# -------------------------------------------------------------------- router ----
class Router(nn.Module):
    def __init__(self, name, K, freeze):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.freeze = freeze
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad_(False)
        else:
            self.enc.gradient_checkpointing_enable()               # bound activation memory when fine-tuning
            self.enc.config.use_cache = False
        d = self.enc.config.hidden_size
        self.policy = nn.Linear(d, K + 1)                          # arm 0 = decline
        self.value = nn.Linear(d, 1)

    def pooled(self, ids, mask):
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return p.detach() if self.freeze else p


# --------------------------------------------------------------------- train ----
def phase_train(base_cfg, pb, model, tok, rm, rm_tok, arm, device):
    O = _out(pb); base_ref = {int(k): v for k, v in json.load(open(O / "base_ref.json")).items()}
    P, bnds = _all_prompts(pb); n_tr, n_ev, _ = bnds
    moves = _moves(pb); K = len(moves); bc = pb["bandit"]
    gcfg = _gcfg(base_cfg, pb["pool"])
    finetune = (arm == "finetune")
    torch.manual_seed(pb["optim"]["seed"])

    etok = AutoTokenizer.from_pretrained(bc["encoder"])
    net = Router(bc["encoder"], K, freeze=not finetune).to(device)
    groups = [{"params": list(net.policy.parameters()) + list(net.value.parameters()), "lr": bc["lr_head"]}]
    if finetune:
        groups.append({"params": net.enc.parameters(), "lr": bc["lr_enc"]})
    opt = torch.optim.AdamW(groups, weight_decay=bc["weight_decay"])

    def enc_batch(prompts):
        e = etok(prompts, padding=True, truncation=True, max_length=bc["max_prompt_tokens"],
                 return_tensors="pt").to(device)
        return e["input_ids"], e["attention_mask"]

    gen_micro = bc.get("gen_micro", 16)                           # 7B generation group (no_grad); OOM was roberta backward, now checkpointed
    def gen_rewards(gis, arms):                                    # only arms>0 generate; decline=0
        r = np.zeros(len(gis))
        for a in sorted(set(int(x) for x in arms if x > 0)):
            idx = [k for k, aa in enumerate(arms) if int(aa) == a]
            for s in range(0, len(idx), gen_micro):
                sub = idx[s:s + gen_micro]
                comps = generate_batch(model, tok, [P[gis[k]] for k in sub], gcfg, system=moves[a - 1])
                for k, c in zip(sub, comps):
                    r[k] = (rm_score(rm, rm_tok, P[gis[k]], c) - base_ref[gis[k]]) if c.strip() else 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                              # release generation KV before roberta backward
        return r

    tr_gis = [gi for gi in range(n_tr) if base_ref.get(gi) is not None]
    rng = np.random.default_rng(pb["optim"]["seed"])
    # --- resume: per-epoch checkpoint so a wall-kill never loses the run (just resubmit to continue) ---
    ckpt_path = O / f"ckpt_{arm}.pt"
    start_ep, curve = 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]; start_ep = ck["epoch"]; curve = ck["curve"]
        print(f"[{arm}] RESUME from epoch {start_ep}/{bc['epochs']} (checkpoint found)", flush=True)
    t0 = time.time()
    for ep in range(start_ep, bc["epochs"]):
        net.train() if finetune else net.eval()
        order = np.array(tr_gis); rng.shuffle(order)
        ep_r, ep_ent, usage = [], [], np.zeros(K + 1)
        for s in range(0, len(order), bc["batch"]):
            gis = list(order[s:s + bc["batch"]])
            ids, mask = enc_batch([P[gi] for gi in gis])
            pooled = net.pooled(ids, mask)
            logits = net.policy(pooled)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            rewards = torch.tensor(gen_rewards(gis, a.tolist()), dtype=torch.float32, device=device)
            v = net.value(pooled).squeeze(1)
            adv = (rewards - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            loss = (-(adv * dist.log_prob(a)).mean() - bc["entropy_beta"] * dist.entropy().mean()
                    + bc["value_coef"] * F.mse_loss(v, rewards))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_r += rewards.tolist(); ep_ent.append(float(dist.entropy().mean().detach()))
            for aa in a.tolist():
                usage[int(aa)] += 1
        curve.append({"epoch": ep + 1, "train_reward": float(np.mean(ep_r)),
                      "policy_entropy": float(np.mean(ep_ent)),
                      "move_usage": (usage / usage.sum()).round(3).tolist(),
                      "wall_s": round(time.time() - t0, 1)})
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": ep + 1, "curve": curve}, ckpt_path)
        print(f"[{arm}] epoch {ep+1}/{bc['epochs']}  train_reward {np.mean(ep_r):+.3f}  "
              f"entropy {np.mean(ep_ent):.3f}  usage {(usage/usage.sum()).round(2).tolist()}", flush=True)

    # router-eval on the 600 held-out eval prompts (argmax policy, 1 generation each)
    net.eval()
    ev_gis = [gi for gi in range(n_tr, n_tr + n_ev) if base_ref.get(gi) is not None]
    eval_by_gi = {}
    with torch.no_grad():
        for s in range(0, len(ev_gis), bc["batch"]):
            gis = ev_gis[s:s + bc["batch"]]
            ids, mask = enc_batch([P[gi] for gi in gis])
            a = net.policy(net.pooled(ids, mask)).argmax(1).tolist()
            r = gen_rewards(gis, a)
            for gi, aa, rr in zip(gis, a, r):
                eval_by_gi[int(gi)] = float(rr if aa > 0 else 0.0)
    res = {"arm": arm, "eval_by_gi": eval_by_gi,
           "eval_mean": float(np.mean(list(eval_by_gi.values()))),
           "eval_ci": _boot(list(eval_by_gi.values())), "curve": curve,
           "trainable_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad))}
    json.dump(res, open(O / f"train_{arm}.json", "w"))
    if ckpt_path.exists():
        ckpt_path.unlink()                                        # training+eval complete; drop the resume checkpoint
    print(f"[{arm}] router-eval ΔRM {res['eval_mean']:+.3f}  -> {O/f'train_{arm}.json'}", flush=True)


# -------------------------------------------------------------------- report ----
def phase_report(pb):
    O = _out(pb); oc = json.load(open(O / "oracle.json"))
    single_by = {int(k): v for k, v in oc["single_by_gi"].items()}
    single = float(oc["single_mean"])
    rows = [f"# B1 online bandit router — {pb['tag']}: FROZEN-head vs FINE-TUNED-encoder ({pb['bandit']['encoder']})\n",
            f"Controlled single-turn test at n_train={pb['n_train']} on the fixed 8-move basis. Online hardened "
            f"REINFORCE, reward = RM(move) - base_ref, decline arm = 0. Router-eval on {oc['n_single']} held-out "
            f"prompts (1 gen each). single {single:+.3f} {tuple(round(x,3) for x in oc['single_ci'])}; de-biased "
            f"oracle {oc['oracle_mean']:+.3f} {tuple(round(x,3) for x in oc['oracle_ci'])} (n={oc['n_ceiling']}).\n",
            "| arm | eval ΔRM | vs single | vs single (paired) | trainable params | final entropy |",
            "|---|---|---|---|---|---|"]
    arms = {}
    for arm in ("frozen", "finetune"):
        p = O / f"train_{arm}.json"
        if not p.exists():
            rows.append(f"| {arm} | (missing) | | | | |"); continue
        r = json.load(open(p)); arms[arm] = r
        eb = {int(k): v for k, v in r["eval_by_gi"].items()}
        gis = [g for g in eb if g in single_by]
        paired = np.array([eb[g] - single_by[g] for g in gis])
        plo, phi = _boot(paired)
        ent = r["curve"][-1]["policy_entropy"]
        rows.append(f"| {arm} | {r['eval_mean']:+.3f} {tuple(round(x,3) for x in r['eval_ci'])} | "
                    f"{r['eval_mean']-single:+.3f} | {paired.mean():+.3f} [{plo:+.3f}, {phi:+.3f}] | "
                    f"{r['trainable_params']:,} | {ent:.3f} |")
    # finetune vs frozen (paired) — the headline
    verdict = "inconclusive (an arm is missing)"
    if "frozen" in arms and "finetune" in arms:
        ef = {int(k): v for k, v in arms["frozen"]["eval_by_gi"].items()}
        eft = {int(k): v for k, v in arms["finetune"]["eval_by_gi"].items()}
        gis = [g for g in ef if g in eft]
        d = np.array([eft[g] - ef[g] for g in gis]); dlo, dhi = _boot(d)
        ceil_break = arms["finetune"]["eval_mean"] > 0.40
        rows += ["", "## Headline — fine-tune vs frozen (paired, same eval prompts)",
                 f"- fine-tune − frozen = {d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]; fine-tune eval {arms['finetune']['eval_mean']:+.3f}.",
                 f"- clears the ~+0.40 single-turn ceiling: {'YES' if ceil_break else 'no'}."]
        if dlo > 0 and ceil_break:
            verdict = ("fine-tune-at-scale BEATS frozen AND clears +0.40 ⇒ single-turn conditioning is ALIVE "
                       "⇒ iterate (more prompts / larger encoder), do not pivot yet.")
        elif dlo > 0:
            verdict = ("fine-tune beats frozen but stays below +0.40 ⇒ real but small gain; the ~+0.38 bound holds "
                       "in magnitude ⇒ document and pivot to Study 2.")
        else:
            verdict = ("fine-tune ≈/below frozen (paired CI includes 0) ⇒ the one untested lever does NOT exceed the "
                       "frozen ceiling ⇒ single-turn conditioning BOUND confirmed ⇒ pivot to Study 2 (bandit carries over).")
    # anti-collapse (rule 4)
    rows += ["", "## Anti-collapse guard (rule 4)"]
    for arm, r in arms.items():
        u = r["curve"][-1]["move_usage"]
        rows.append(f"- {arm}: final move-usage {u} (arm0=decline); entropy {r['curve'][-1]['policy_entropy']:.3f} "
                    f"⇒ {'OK (spread)' if max(u) < 0.8 else '⚠ COLLAPSED to one move'}.")
    rows += ["", "## Verdict", f"- {verdict}",
             "", "## Cost", *[f"- {arm}: {json.load(open(O/f'train_{arm}.json'))['curve'][-1]['wall_s']/3600:.2f} "
                              f"GPU-h train, {json.load(open(O/f'train_{arm}.json'))['trainable_params']:,} trainable params"
                              for arm in arms]]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_bandit_online_{pb['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(x for x in rows if not x.startswith("|")))
    print(f"\nreport -> {rpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["pool", "assemble", "prep", "assemble_prep", "train", "report"])
    ap.add_argument("--arm", choices=["frozen", "finetune"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    args = ap.parse_args()
    base_cfg = load_config(args.base_config); pb = _pb(args.config)
    device = resolve_device(base_cfg)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    t0 = time.time()
    needs_gen = args.phase in ("pool", "prep", "train")
    model = tok = rm = rm_tok = None
    if needs_gen:
        model, tok = load_base(base_cfg, device)
        rm, rm_tok = load_rm(base_cfg, device)

    if args.phase == "pool":
        phase_pool(base_cfg, pb, model, tok, rm, rm_tok, shard)
    elif args.phase == "assemble":
        assemble_pool(pb)
    elif args.phase == "prep":
        phase_prep(base_cfg, pb, model, tok, rm, rm_tok, shard)
    elif args.phase == "assemble_prep":
        assemble_prep(pb)
    elif args.phase == "train":
        assert args.arm, "--arm frozen|finetune required"
        phase_train(base_cfg, pb, model, tok, rm, rm_tok, args.arm, device)
    elif args.phase == "report":
        phase_report(pb)
    if needs_gen:
        print(log_cost("B1", f"bandit_online_{args.phase}" + (f"_{args.arm}" if args.arm else ""),
                       time.time() - t0, device, notes="online contextual-bandit router (no 7B backprop)"))


if __name__ == "__main__":
    main()
