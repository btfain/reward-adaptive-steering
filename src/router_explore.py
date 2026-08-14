"""Offline router study — iterate router architectures in seconds against the CACHED read states
(results/prompt_basis_<tag>/states.npz) + the train swing matrix (swing_train.npz). NO model, NO
generation, minimal deps (numpy + torch), so it runs fast on a login node / laptop.

Addresses the scaled-run finding (learned router ≈ best single move despite a big de-biased oracle
gap). All requested changes:
  * VALUE-REGRESSION router: predict the swing vector ŝ(h)∈R^K and argmax over moves — uses all
    swing info and handles near-ties, unlike the hard argmax classifier.
  * proper 3-way split (router-train / val / eval) with EARLY STOPPING (the in-run router had none).
  * reduced capacity + dropout + stronger weight-decay knobs (the MLP overfit: train acc 1.0).
  * HONEST selection: pick (layer, variant, hp) by VAL realized-ΔRM, report EVAL realized-ΔRM.

Eval is on held-out TRAIN prompts scored by the train swings (m_swing) — noisier than the full run's
m_test de-biased oracle (~+0.81 there), but it needs no generation, so it compares architectures
freely. If a variant clearly beats the single move here, confirm on the honest test set next.

    python src/router_explore.py --tag large_7b      # run on the cluster login node (CPU is fine)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


def _labels(Msel):
    y = []
    for r in Msel:
        if np.all(np.isnan(r)):
            y.append(0); continue
        j = int(np.nanargmax(r))
        y.append(0 if np.nan_to_num(r[j], nan=-1e9) <= 0 else j + 1)
    return np.array(y)


def _pca(Hfit, Hs, k):
    mu = Hfit.mean(0)
    _, _, Vt = np.linalg.svd(Hfit - mu, full_matrices=False)
    C = Vt[:k]
    return [(H - mu) @ C.T for H in Hs]


def _fit(Ztr, Ttr, Zval, Tval, out_dim, reg, mlp, hp, device, seed=0):
    torch.manual_seed(seed)
    din = Ztr.shape[1]
    mods = ([nn.Linear(din, hp["hidden"]), nn.ReLU(), nn.Dropout(hp["dropout"]),
             nn.Linear(hp["hidden"], out_dim)] if mlp else [nn.Linear(din, out_dim)])
    net = nn.Sequential(*mods).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    Ztr_t = torch.tensor(Ztr, dtype=torch.float32, device=device)
    Zval_t = torch.tensor(Zval, dtype=torch.float32, device=device)
    if reg:
        A = torch.tensor(np.nan_to_num(Ttr), dtype=torch.float32, device=device)
        Am = torch.tensor(~np.isnan(Ttr), dtype=torch.float32, device=device)
        B = torch.tensor(np.nan_to_num(Tval), dtype=torch.float32, device=device)
        Bm = torch.tensor(~np.isnan(Tval), dtype=torch.float32, device=device)
        ltr = lambda p: ((p - A) ** 2 * Am).sum() / Am.sum().clamp(min=1)
        lval = lambda p: float((((p - B) ** 2 * Bm).sum() / Bm.sum().clamp(min=1)).item())
    else:
        yt = torch.tensor(Ttr, dtype=torch.long, device=device)
        yv = torch.tensor(Tval, dtype=torch.long, device=device)
        ltr = lambda p: F.cross_entropy(p, yt)
        lval = lambda p: float(F.cross_entropy(p, yv).item())
    best, best_state, bad = 1e18, None, 0
    for _ in range(hp["epochs"]):
        net.train(); opt.zero_grad(); ltr(net(Ztr_t)).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = lval(net(Zval_t))
        if v < best - 1e-5:
            best, best_state, bad = v, {k: p.detach().clone() for k, p in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    net.load_state_dict(best_state)
    return net


def _pred(net, Z, reg, device):
    with torch.no_grad():
        out = net(torch.tensor(Z, dtype=torch.float32, device=device)).cpu().numpy()
    if reg:
        j = out.argmax(1)
        return np.array([0 if out[i, j[i]] <= 0 else j[i] + 1 for i in range(len(out))])
    return out.argmax(1)


def _realized(pred, Msel):
    return np.array([0.0 if pred[i] == 0 else np.nan_to_num(Msel[i, pred[i] - 1], nan=0.0)
                     for i in range(len(pred))])


def _boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    b = np.sort(v[rng.integers(0, len(v), (n, len(v)))].mean(1))
    return float(b[int(0.025 * n)]), float(b[int(0.975 * n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--n_pca", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    st = np.load(OUT / "states.npz", allow_pickle=True)
    layers = [int(L) for L in st["layers"]]
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]                                        # (n_train, K) train swings over selected moves
    K = len(S)
    ok = ~np.isnan(Msel).all(1)
    idx = np.where(ok)[0]
    rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
    tr_i, va_i, ev_i = idx[:a], idx[a:b], idx[b:]              # router-train / val / eval prompts
    ycls = _labels(Msel)

    single = np.nan_to_num(Msel[ev_i, 0], nan=0.0)
    naive_or = np.array([max(0.0, np.nan_to_num(Msel[i], nan=-1e9).max()) for i in ev_i])
    slo, shi = _boot(single); olo, ohi = _boot(naive_or)

    grid_mlp = [dict(hidden=h, dropout=d, lr=0.05, wd=wd, epochs=800, patience=40)
                for h in (32, 64) for d in (0.3, 0.5) for wd in (0.01, 0.1)]
    grid_lin = [dict(hidden=0, dropout=0.0, lr=0.05, wd=wd, epochs=800, patience=40) for wd in (0.01, 0.1)]

    rows, best = [], None
    for L in layers:
        Htr = st[f"Htr_{L}"]
        Ztr_all, Zva_all, Zev_all = _pca(Htr[tr_i], [Htr[tr_i], Htr[va_i], Htr[ev_i]], args.n_pca)
        for vname, reg in (("cls", False), ("reg", True)):
            for cname, mlp in (("linear", False), ("mlp", True)):
                out_dim = K if reg else K + 1
                Ttr = Msel[tr_i] if reg else ycls[tr_i]
                Tval = Msel[va_i] if reg else ycls[va_i]
                for hp in (grid_mlp if mlp else grid_lin):
                    net = _fit(Ztr_all, Ttr, Zva_all, Tval, out_dim, reg, mlp, hp, device, args.seed)
                    val_r = _realized(_pred(net, Zva_all, reg, device), Msel[va_i]).mean()
                    ev = _realized(_pred(net, Zev_all, reg, device), Msel[ev_i])
                    rec = {"layer": L, "variant": vname, "cap": cname, "hp": hp,
                           "val": float(val_r), "eval": float(ev.mean()), "ev_arr": ev}
                    rows.append(rec)
                    if best is None or rec["val"] > best["val"]:      # SELECT ON VAL
                        best = rec
    fam = {}
    for r in rows:
        key = f"{r['variant']}_{r['cap']}"
        if key not in fam or r["val"] > fam[key]["val"]:
            fam[key] = r
    rlo, rhi = _boot(best["ev_arr"])

    L = [f"# Router study (offline) — {args.tag}: value-regression + early-stopping + honest selection\n",
         f"Cached states {OUT}/states.npz + train swings. {n} valid prompts "
         f"({len(tr_i)} router-train / {len(va_i)} val / {len(ev_i)} eval), K={K} moves, PCA-{args.n_pca}, "
         f"layers {layers}. Selected on VAL realized-ΔRM, reported on held-out EVAL (train swings, m_swing — "
         f"noisier than the run's m_test de-biased oracle ~+0.81).\n",
         f"Eval baselines: single move {single.mean():+.3f} [{slo:+.3f}, {shi:+.3f}]; naive(biased) oracle "
         f"{naive_or.mean():+.3f} [{olo:+.3f}, {ohi:+.3f}].\n",
         "| variant | cap | best layer | val ΔRM | **eval ΔRM** |", "|---|---|---|---|---|"]
    for key in ("cls_linear", "cls_mlp", "reg_linear", "reg_mlp"):
        if key in fam:
            r = fam[key]
            L.append(f"| {r['variant']} | {r['cap']} | {r['layer']} | {r['val']:+.3f} | **{r['eval']:+.3f}** |")
    beats = "BEATS" if best["eval"] > shi else ("≈" if best["eval"] > single.mean() - 0.02 else "below")
    L += ["", f"## Best (val-selected): {best['variant']}/{best['cap']} @ layer {best['layer']}  hp={best['hp']}",
          f"- **eval ΔRM {best['eval']:+.3f} [{rlo:+.3f}, {rhi:+.3f}]** vs single move {single.mean():+.3f}  ({beats} single).",
          "", "## Reading",
          "- **reg beats cls and clears single (eval CI vs single)** ⇒ conditioning IS extractable — the hard "
          "argmax classifier / overfitting was the problem, not the signal. Confirm on the honest test set.",
          "- **all variants ≈ single** ⇒ h→move is genuinely hard here ⇒ clean Subproject-1 negative on "
          "extraction; carry routing to multi-turn (where the 'which move' signal should be far stronger)."]
    BASIS.mkdir(exist_ok=True)
    (BASIS / f"s1_router_{args.tag}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(x for x in L if not x.startswith("|")))
    print(f"\nreport -> {BASIS / f's1_router_{args.tag}_report.md'}")


if __name__ == "__main__":
    main()
