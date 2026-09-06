"""Step 1 of the RM-noise diagnostic — calibrate the reward model's NOISE SCALE against ground-truth
preference labels (NOT against a second RM). RM preference accuracy isn't a proxy for BoN selection quality
— it IS the primitive BoN runs on (BoN-2 = pick the better of 2 same-prompt samples). So the RM's accuracy
on same-prompt preference pairs, measured vs the dataset's labels, bounds how often BoN picks the truly
better sample, and the residual pins the noise scale sigma_eps in native RM units.

Model per pair: RM = (signal aligned with the label) + eps.  Label margin m = score_chosen - score_rejected.
  * accuracy-vs-margin  A(m) = P(RM_chosen > RM_rejected | m)   — the direct, interpretable noise gauge.
  * near-tied pairs (m ~ 0 => true Δreward ~ 0): RM_chosen - RM_rejected ~ eps_c - eps_r, so
      sigma_eps^2 ~= 1/2 * Var(ΔRM | smallest-margin band),  cross-checked by the residual of ΔRM ~ m.
sigma_eps (RM units) then feeds Step 2: real fraction of BoN headroom ~= 1 - sigma_eps^2 / sigma_s^2, where
sigma_s^2 is the within-prompt RM-score variance measured on a generation pool.

RM_B (the alt RM) is scored too but ONLY as a secondary descriptive cross-check — the ground truth here is
the preference labels, not RM_B.

  # cluster (RM forward only, no generation):
  python src/rm_pref_noise.py --phase score --n 4000
  # local (offline):
  python src/rm_pref_noise.py --phase analyze
"""

import argparse
import time
from pathlib import Path

import numpy as np

from models import REPO_ROOT, load_config, load_rm, log_cost, resolve_device, rm_score

OUT = REPO_ROOT / "results" / "rm_pref_noise"


def _resp(msgs):
    """UF 'chosen'/'rejected' are message lists; the response is the last assistant turn."""
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1].get("content", "")
    return msgs if isinstance(msgs, str) else ""


def phase_score(cfg, n, seed):
    from datasets import load_dataset
    device = resolve_device(cfg); t0 = time.time()
    ds = load_dataset(cfg["data"]["prompt_dataset"], split="train_prefs")
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    rm_a, tok_a = load_rm(cfg, device)
    cfg_b = dict(cfg); cfg_b["reward_model"] = cfg["reward_model_alt"]
    try:
        rm_b, tok_b = load_rm(cfg_b, device)
    except Exception as e:
        print(f"RM_B load failed ({e}); scoring RM_A only", flush=True); rm_b = None
    rows = {k: [] for k in ("sa_c", "sa_r", "sb_c", "sb_r", "m_c", "m_r")}
    kept = 0
    for j, i in enumerate(idx):
        ex = ds[int(i)]
        pr = ex["prompt"]; yc = _resp(ex["chosen"]); yr = _resp(ex["rejected"])
        sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
        if not (yc.strip() and yr.strip()) or sc is None or sr is None:
            continue
        try:
            rows["sa_c"].append(rm_score(rm_a, tok_a, pr, yc))
            rows["sa_r"].append(rm_score(rm_a, tok_a, pr, yr))
            rows["sb_c"].append(rm_score(rm_b, tok_b, pr, yc) if rm_b else np.nan)
            rows["sb_r"].append(rm_score(rm_b, tok_b, pr, yr) if rm_b else np.nan)
        except Exception:
            continue
        rows["m_c"].append(float(sc)); rows["m_r"].append(float(sr)); kept += 1
        if (j + 1) % 500 == 0:
            print(f"  scored {j+1}/{len(idx)} (kept {kept})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "scores.npz", **{k: np.array(v, float) for k, v in rows.items()},
             rm_a=cfg["reward_model"], rm_b=cfg["reward_model_alt"])
    print(log_cost("RBnoise", "pref_score", time.time() - t0, device, notes=f"{kept} pairs, dual-RM"))
    print(f"scored {kept} pairs -> {OUT/'scores.npz'}")


def _bands(m):
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 12.0]
    return edges


def phase_analyze():
    d = np.load(OUT / "scores.npz", allow_pickle=True)
    m_raw = d["m_c"] - d["m_r"]                               # label margin (>=0 by construction)
    dA_raw = d["sa_c"] - d["sa_r"]                            # RM_A score difference (chosen - rejected)
    dB_raw = d["sb_c"] - d["sb_r"]                            # RM_B (secondary), may be all-nan
    ok = np.isfinite(m_raw) & np.isfinite(dA_raw)
    m, dA = m_raw[ok], dA_raw[ok]
    accA = float((dA > 0).mean())

    # accuracy + ΔRM variance per margin band
    edges = _bands(m); rows_band = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m >= lo) & (m < hi)
        if sel.sum() < 10:
            continue
        rows_band.append((lo, hi, int(sel.sum()), float(m[sel].mean()),
                          float((dA[sel] > 0).mean()), float(np.var(dA[sel]))))

    # sigma_eps: near-tied band (smallest) gives 1/2 Var(ΔRM); residual-of-ΔRM~m gives a binning-free check
    tied = m < 0.5
    sig_band = np.sqrt(0.5 * np.var(dA[tied])) if tied.sum() >= 20 else float("nan")
    A = np.vstack([m, np.ones_like(m)]).T
    coef, *_ = np.linalg.lstsq(A, dA, rcond=None)
    resid = dA - A @ coef
    sig_resid = float(np.sqrt(0.5 * np.var(resid)))
    r2 = 1.0 - np.var(resid) / np.var(dA)                    # frac of ΔRM variance aligned with the label

    # scale context: spread of RM scores across responses (chosen+rejected pooled)
    pooled = np.concatenate([d["sa_c"][np.isfinite(d["sa_c"])], d["sa_r"][np.isfinite(d["sa_r"])]])
    sd_pooled = float(np.std(pooled))

    lines = ["# Step 1 — RM noise scale from preference labels (ground truth = dataset labels, NOT a 2nd RM)\n",
             f"RM_A = {d['rm_a']}. {len(m)} UF-binarized pairs. Label margin m = score_chosen - score_rejected. "
             "ΔRM = RM_A(chosen) - RM_A(rejected). RM preference accuracy = the exact BoN-2 selection primitive.\n",
             f"- **Overall preference accuracy A = {accA:.3f}** (RM ranks the labeled-better response above the "
             f"worse one {accA*100:.1f}% of the time).",
             "",
             "## Accuracy & ΔRM spread vs label margin",
             "| margin band | n | mean m | accuracy | Var(ΔRM) |", "|---|---|---|---|---|"]
    for lo, hi, nsel, mm, a, v in rows_band:
        lines.append(f"| [{lo:g}, {hi:g}) | {nsel} | {mm:.2f} | {a:.3f} | {v:.3f} |")
    lines += ["",
              "## Noise scale sigma_eps (native RM units)",
              f"- near-tied band (m<0.5): **sigma_eps ~= {sig_band:.3f}**  (= sqrt(1/2 Var(ΔRM | tied)))",
              f"- residual of ΔRM ~ m (binning-free): **sigma_eps ~= {sig_resid:.3f}**  (slope {coef[0]:.3f} RM/label-pt)",
              f"- fraction of ΔRM variance aligned with the label (R^2) = {r2:.3f} "
              f"=> {(1-r2)*100:.0f}% of RM-difference variance is NOT label-aligned (noise + label-noise + nonlinearity).",
              f"- context: pooled RM-score SD across responses = {sd_pooled:.3f} (cross-response, not within-prompt).",
              "",
              "## Reading",
              f"- A={accA:.3f} is the BoN-2 selection reliability against ground-truth labels: ~{(1-accA)*100:.0f}% "
              "of the time BoN-2 keeps the truly-worse sample. If A on CLEAR pairs (large m) is still well below "
              "1.0, that shortfall is gross RM noise that even a big true gap can't overcome.",
              "- sigma_eps here (RM units) feeds Step 2: real fraction of BoN headroom ~= 1 - sigma_eps^2/sigma_s^2, "
              "with sigma_s^2 = within-prompt RM-score variance from a generation pool (next).",
              "- CAVEAT: labels are the dataset's GPT-4 annotations (independent of our RM, but not human) and this "
              "sigma_eps is IN-distribution => an OPTIMISTIC (lower) bound on noise for off-distribution BoN winners."]

    # secondary, descriptive only: cross-RM agreement (NOT ground truth)
    okb = np.isfinite(m_raw) & np.isfinite(dA_raw) & np.isfinite(dB_raw)
    if okb.sum() > 50:
        accB = float((dB_raw[okb] > 0).mean())
        agree = float(((dA_raw[okb] > 0) == (dB_raw[okb] > 0)).mean())
        lines += ["", "## Secondary cross-check (descriptive, NOT ground truth)",
                  f"- RM_B ({d['rm_b']}) accuracy vs labels = {accB:.3f} on {int(okb.sum())} pairs; "
                  f"RM_A/RM_B argmax agreement = {agree:.3f}."]

    OUT.mkdir(parents=True, exist_ok=True)
    rpt = REPO_ROOT / "basis" / "rb_pref_noise_report.md"
    rpt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nreport -> {rpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["score", "analyze"])
    ap.add_argument("--base-config", default="configs/base_7b.yaml")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "score":
        phase_score(load_config(args.base_config), args.n, args.seed)
    else:
        phase_analyze()


if __name__ == "__main__":
    main()
