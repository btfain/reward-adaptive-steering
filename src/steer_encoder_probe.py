"""FREE go/no-go for the encoder-as-steering-controller idea. steer_reach found per-prompt optimal
steering δ's (deltas.npz: D) that carry real reward headroom (+0.49 oracle) but were NOT predictable
from the LLM's OWN read state h(x) (held-out R²≈-0.21) — the "not controllable" wall. We never tried a
DEDICATED ENCODER, the exact representation that rescued the prompt-move router. This regresses
encoder(prompt) → δ and compares held-out DIRECTION recovery (cosine) + R² against the h→δ baseline,
on the SAME 120/50 split. NO generation, NO training of the 7B — CPU, minutes.

  encoder cosine ≫ h cosine (and > 0)  ⇒ the encoder sees steering-relevant structure h didn't ⇒ an
                                          encoder→δ controller is worth building (GPU arm justified).
  encoder ≈ h ≈ 0                       ⇒ δ isn't in the prompt text either ⇒ steering stays dead;
                                          zero GPU spent to learn it.

    python src/steer_encoder_probe.py --tag detrunc_7b --encoders roberta-base,intfloat/e5-large-v2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from steer_cond import _prompts

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"

# steer_reach prompt counts per run tag (must match the config the deltas were produced with)
COUNTS = {"detrunc_7b": (120, 50), "7b": (150, 50)}


def embed(encoder, prompts, batch=16):
    tok = AutoTokenizer.from_pretrained(encoder)
    mdl = AutoModel.from_pretrained(encoder).eval()
    prefix = "query: " if "e5" in encoder.lower() else ""
    out = []
    with torch.no_grad():
        for s in range(0, len(prompts), batch):
            e = tok([prefix + p for p in prompts[s:s + batch]], padding=True, truncation=True,
                    max_length=160, return_tensors="pt")
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().numpy())
    return np.concatenate(out)


def ridge_probe(X, D, tr, te, lam=1.0):
    """Ridge X->D on train; held-out aggregate R² + mean per-prompt cosine(pred, true)."""
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    dmu = D[tr].mean(0)
    Dtr = D[tr] - dmu
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Dtr)                       # (feat, 3584)
    pred = Xte @ W + dmu
    true = D[te]
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    cos = float(np.mean([np.dot(pred[i], true[i]) /
                         (np.linalg.norm(pred[i]) * np.linalg.norm(true[i]) + 1e-9)
                         for i in range(len(te))]))
    return r2, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="detrunc_7b")
    ap.add_argument("--encoders", default="roberta-base,intfloat/e5-large-v2")
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()
    OUT = REPO_ROOT / "results" / f"steer_reach_{args.tag}"
    z = np.load(OUT / "deltas.npz", allow_pickle=True)
    D, H, split = z["D"], z["H"], z["split"]                  # D,H: (N,3584); split: True=train
    n_tr, n_te = COUNTS[args.tag]
    P = _prompts(n_tr, n_te)
    prompts = list(P["train"]) + list(P["test"])
    assert len(prompts) == len(D), f"prompt reconstruction mismatch: {len(prompts)} vs {len(D)} rows"
    # sanity: the split mask should be n_tr True then n_te False (items order)
    assert split[:n_tr].all() and not split[n_tr:].any(), "split mask != reconstructed order"
    tr, te = np.where(split)[0], np.where(~split)[0]

    rows = [f"# Encoder-as-steering-controller probe (offline, free) — steer_reach_{args.tag}\n",
            f"Regress features → per-prompt optimal δ (deltas.npz, {D.shape[0]}×{D.shape[1]}), ridge λ={args.lam}, "
            f"held-out {len(te)} test prompts. Metric = direction recovery (mean cosine) + aggregate R². "
            f"Baseline = the LLM's own read state h(x) (steer_reach's 'not controllable' finding).\n",
            "| features | dim | held-out cosine(pred,true δ) | held-out R² |", "|---|---|---|---|"]
    hr2, hcos = ridge_probe(H, D, tr, te, args.lam)
    rows.append(f"| h(x) LLM read state | {H.shape[1]} | {hcos:+.3f} | {hr2:+.3f} |")
    enc_best = None
    for enc in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        try:
            X = embed(enc, prompts)
        except Exception as e:
            rows.append(f"| {enc} | — | SKIP ({str(e)[:40]}) | |"); continue
        r2, cos = ridge_probe(X, D, tr, te, args.lam)
        rows.append(f"| {enc} | {X.shape[1]} | {cos:+.3f} | {r2:+.3f} |")
        if enc_best is None or cos > enc_best[1]:
            enc_best = (enc, cos, r2)

    rows += ["", "## Reading"]
    if enc_best:
        gain = enc_best[1] - hcos
        if enc_best[1] > 0.1 and gain > 0.05:
            rows.append(f"- **{enc_best[0]} recovers δ-direction (cosine {enc_best[1]:+.3f}) far better than h "
                        f"({hcos:+.3f}), Δ{gain:+.3f}** ⇒ the encoder sees steering structure the LLM state didn't "
                        "⇒ an encoder→δ controller is worth building; plan the GPU arm (gated by B1).")
        elif enc_best[1] > 0.1:
            rows.append(f"- {enc_best[0]} cosine {enc_best[1]:+.3f} > 0 but ≈ h ({hcos:+.3f}) ⇒ modest, not clearly "
                        "better than the LLM state ⇒ weak case; revisit only if B1 also motivates single-turn.")
        else:
            rows.append(f"- **{enc_best[0]} cosine {enc_best[1]:+.3f} ≈ 0 ≈ h ({hcos:+.3f})** ⇒ the per-prompt steering "
                        "direction is NOT predictable from the prompt text by any representation ⇒ steering stays "
                        "walled; do NOT spend GPU. Consistent with the coverage-wall diagnosis.")
    rows.append("- Direction (cosine) is the controllability signal — magnitude is capped in steering; R² over 3584 "
                "dims is naturally low, so read the h-vs-encoder DELTA, not the absolute.")
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_steer_encoder_probe_{args.tag}_report.md"
    rpt.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
