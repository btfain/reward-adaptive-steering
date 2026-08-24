"""Step 1 — stronger ranker (FREE, no new generation). The bake-off used a ridge/distilroberta router
and got router-move best-of-2 +0.99 vs oracle-2 +1.53 — i.e. it realized only part of the ranking
headroom. Here we re-rank the SAME cached generations with several rankers and see how much of the
router→oracle gap each closes. Only the RANKING varies; naive/oracle/random curves are fixed.

Rankers (all trained on large_7b swing matrix, applied to the held-out bake-off prompts):
  ridge_distil  frozen distilroberta + ridge   (= the bake-off baseline)
  ridge_e5      frozen e5-large-v2 + ridge      (stronger frozen representation)
  mlp_distil    frozen distilroberta + value-regression MLP (more capacity, early-stopped)

    python src/bakeoff_rankers.py --config configs/bakeoff_7b.yaml
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from bakeoff import _cfg, _out, _prompts, _moves, _emax_k, REPO_ROOT


def embed(encoder, texts, max_len=160, batch=16):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder); mdl = AutoModel.from_pretrained(encoder).eval()
    pre = "query: " if "e5" in encoder.lower() else ""
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            e = tok([pre + t for t in texts[s:s + batch]], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            h = mdl(**e).last_hidden_state; m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
    return np.concatenate(out)


def _l7(c):
    """large_7b train prompts + swing targets over the selected moves (ranker training set)."""
    S = json.load(open(REPO_ROOT / c["basis_selection"]))["order"]
    M = np.load(REPO_ROOT / c["router_swing"], allow_pickle=True)["M"][:, S]
    pb = __import__("yaml").safe_load(open(REPO_ROOT / "configs" / "prompt_basis_large_7b.yaml"))
    allp = json.load(open(REPO_ROOT / "data" / "prompts.json"))[pb.get("prompts_split", "train")]
    P = allp[:pb["pool"]["n_prompts_train"]]
    ok = ~np.isnan(M).all(1)
    return [P[i] for i in np.where(ok)[0]], np.nan_to_num(M[ok])


def _ridge_rank(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = (Xtr - mu) / sd; dmu = Ytr.mean(0)
    W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ (Ytr - dmu))
    return np.argsort(-(((Xte - mu) / sd) @ W + dmu), axis=1)


def _mlp_rank(Xtr, Ytr, Xte, seed=0):
    torch.manual_seed(seed)
    n = len(Xtr); a = int(0.85 * n); idx = np.random.default_rng(seed).permutation(n)
    tr, va = idx[:a], idx[a:]
    mu, sd = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-6
    Z = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); Y = torch.tensor(Ytr, dtype=torch.float32)
    Zte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Z.shape[1], 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.1)
    best, bstate, bad = 1e9, None, 0
    for _ in range(600):
        net.train(); opt.zero_grad()
        nn.functional.mse_loss(net(Z[tr]), Y[tr]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Z[va]), Y[va]))
        if v < best - 1e-5:
            best, bstate, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 40:
                break
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad():
        return np.argsort(-net(Zte).numpy(), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bakeoff_7b.yaml")
    args = ap.parse_args()
    c = _cfg(args.config); O = _out(c)
    shards = sorted(O.glob("gen_shard_*.jsonl")) or [O / "gen.jsonl"]
    base = defaultdict(list); move = defaultdict(lambda: defaultdict(list))
    for sh in shards:
        for l in open(sh):
            r = json.loads(l)
            (base[r["pi"]].append(r["rm"]) if r["kind"] == "base"
             else move[r["pi"]][r["move"]].append(r["rm"]))
    P = _prompts(c); K = len(_moves(c))
    pis = [pi for pi in range(len(P)) if len(base.get(pi, [])) >= 2 and len(move.get(pi, {})) == K]
    prompts = [P[pi] for pi in pis]; nb = min(len(base[pi]) for pi in pis)

    # rankers
    Ptr, Ytr = _l7(c)
    Hd = np.load(REPO_ROOT / c["router_enc_embed"], allow_pickle=True)["Htr"]      # cached distilroberta (large_7b)
    ok = ~np.isnan(np.load(REPO_ROOT / c["router_swing"], allow_pickle=True)["M"]).all(1)
    Hd = Hd[ok]
    Xd_te = embed("distilroberta-base", prompts)
    rankers = {"ridge_distil": _ridge_rank(Hd, Ytr, Xd_te),
               "mlp_distil": _mlp_rank(Hd, Ytr, Xd_te)}
    try:
        He = embed("intfloat/e5-large-v2", Ptr); Xe_te = embed("intfloat/e5-large-v2", prompts)
        rankers["ridge_e5"] = _ridge_rank(He, Ytr, Xe_te)
    except Exception as e:
        print("e5 skipped:", str(e)[:60])

    rng = np.random.default_rng(0); DRAWS = 400
    naive = np.zeros(K + 1); omove = np.zeros(K + 1); rndm = np.zeros(K + 1)
    rmv = {name: np.zeros(K + 1) for name in rankers}
    for r_i, pi in enumerate(pis):
        b = np.array(base[pi]); bref = b.mean()
        sw = {mj: np.array(move[pi][mj]) - bref for mj in range(K)}
        oracle_rank = np.argsort(-np.array([sw[mj].mean() for mj in range(K)]))
        for k in range(1, K + 1):
            naive[k] += _emax_k(b, min(k, nb)) - bref
            omove[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in oracle_rank[:k]) for _ in range(DRAWS)]))
            rndm[k] += float(np.mean([max(rng.choice(sw[mj]) for mj in rng.permutation(K)[:k]) for _ in range(DRAWS)]))
            for name, rk in rankers.items():
                rmv[name][k] += float(np.mean([max(rng.choice(sw[mj]) for mj in rk[r_i][:k]) for _ in range(DRAWS)]))
    n = len(pis)
    for a in [naive, omove, rndm] + list(rmv.values()):
        a /= n

    def frac(x, k):                                # % of ranking headroom (random→oracle) captured at k
        lo, hi = rndm[k], omove[k]
        return (x[k] - lo) / (hi - lo) * 100 if hi > lo else float("nan")

    cols = list(rmv)
    rows = [f"# Bake-off — stronger ranker comparison (FREE, same generations) — {c['tag']}\n",
            f"{n} held-out prompts, best-of-k over the SAME cached gens; only the RANKER varies. "
            f"naive/oracle/random fixed. router = trained on large_7b swings, applied to held-out prompts.\n",
            "| k | naive | " + " | ".join(cols) + " | oracle | random |",
            "|---|" + "---|" * (len(cols) + 3)]
    for k in range(1, K + 1):
        rows.append(f"| {k} | {naive[k]:+.3f} | " + " | ".join(f"{rmv[nm][k]:+.3f}" for nm in cols)
                    + f" | {omove[k]:+.3f} | {rndm[k]:+.3f} |")
    rows += ["", "## At k=2 (the headline budget)"]
    for nm in cols:
        rows.append(f"- **{nm}**: {rmv[nm][2]:+.3f}  (vs naive {naive[2]:+.3f}, Δ{rmv[nm][2]-naive[2]:+.3f}; "
                    f"captures {frac(rmv[nm],2):.0f}% of random→oracle ranking headroom)")
    best = max(cols, key=lambda nm: rmv[nm][2])
    base_r = rmv["ridge_distil"][2]
    rows += ["", "## Reading",
             f"- best ranker at k=2: **{best}** ({rmv[best][2]:+.3f}); vs bake-off ridge_distil "
             f"({base_r:+.3f}) Δ{rmv[best][2]-base_r:+.3f}.",
             ("- a stronger ranker MOVES the k=2 result toward the oracle ⇒ ranker quality is a live lever; "
              "worth a better router." if rmv[best][2] > base_r + 0.03 else
              "- stronger rankers ≈ ridge_distil at k=2 ⇒ ranker quality is NOT the bottleneck here; the ridge "
              "router already captures most reachable ranking signal (consistent with the prediction bound — "
              "the residual gap to oracle is info-limited, not model-limited)."),
             f"- oracle-2 {omove[2]:+.3f} ≈ naive best-of-{next((k for k in range(1,K+1) if naive[k]>=omove[2]), '>'+str(K))} "
             "⇒ the ceiling of router-narrowed selection if the ranker were perfect."]
    rpt = REPO_ROOT / "basis" / f"s1_bakeoff_rankers_{c['tag']}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
