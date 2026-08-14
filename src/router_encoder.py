"""Dedicated-encoder router: fine-tune a small encoder (default distilroberta-base, ~82M) on the
prompt TEXT to predict the swing vector ŝ(x)∈R^K, then argmax over the selected moves. Tests whether
conditioning is extractable by a PURPOSE-TRAINED router over the prompt text — as opposed to the
LLM/generator's own last-token hidden state (optimized for next-token prediction), which failed to
generalize (router_explore: eval ≈ / below single move, flat learning curve).

Same train/val/eval split + honest val-selection as router_explore, so numbers are comparable. The
swing targets come from the completed run's swing_train.npz; no LLM/generation needed here.

    python src/router_encoder.py --tag large_7b                 # fine-tune (GPU, ~minutes)
    python src/router_encoder.py --tag large_7b --freeze        # frozen encoder + linear head (baseline)
    python src/router_encoder.py --tag large_7b --encoder microsoft/deberta-v3-small
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS = REPO_ROOT / "basis"


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


class EncoderRouter(nn.Module):
    def __init__(self, name, K, freeze):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad_(False)
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, K))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean pool
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="large_7b")
    ap.add_argument("--config", default="configs/prompt_basis_large_7b.yaml")
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pb = yaml.safe_load(open(REPO_ROOT / args.config))
    OUT = REPO_ROOT / "results" / f"prompt_basis_{args.tag}"
    prompts_all = json.load(open(REPO_ROOT / "data" / "prompts.json"))[pb.get("prompts_split", "train")]
    prompts = prompts_all[:pb["pool"]["n_prompts_train"]]
    sw = np.load(OUT / "swing_train.npz", allow_pickle=True)
    S = json.load(open(OUT / "selection.json"))["order"]
    Msel = sw["M"][:, S]
    K = len(S)
    ok = ~np.isnan(Msel).all(1)
    idx = np.where(ok)[0]
    rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    n = len(idx); a, b = int(0.6 * n), int(0.8 * n)
    tr_i, va_i, ev_i = idx[:a], idx[a:b], idx[b:]                # SAME split as router_explore

    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = tok([prompts[i] for i in range(len(prompts))], padding=True, truncation=True,
              max_length=160, return_tensors="pt")
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]
    T = torch.tensor(np.nan_to_num(Msel), dtype=torch.float32)
    Tm = torch.tensor(~np.isnan(Msel), dtype=torch.float32)

    net = EncoderRouter(args.encoder, K, args.freeze).to(device)
    groups = [{"params": net.head.parameters(), "lr": 1e-3}]
    if not args.freeze:
        groups.append({"params": net.enc.parameters(), "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    def batches(ix):
        for s in range(0, len(ix), args.batch):
            yield ix[s:s + args.batch]

    def masked_mse(pred, rows):
        t = T[rows].to(device); m = Tm[rows].to(device)
        return ((pred - t) ** 2 * m).sum() / m.sum().clamp(min=1)

    def val_loss():
        net.eval()
        with torch.no_grad():
            p = net(ids_all[va_i].to(device), mask_all[va_i].to(device))
            return float(masked_mse(p, va_i).item())

    best, best_state, bad = 1e18, None, 0
    for ep in range(args.epochs):
        net.train(); order = tr_i.copy(); rng.shuffle(order)
        for rows in batches(order):
            opt.zero_grad()
            p = net(ids_all[rows].to(device), mask_all[rows].to(device))
            masked_mse(p, rows).backward(); opt.step()
        v = val_loss()
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.detach().cpu().clone() for k, p in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)

    def predict(ix):
        net.eval()
        with torch.no_grad():
            p = net(ids_all[ix].to(device), mask_all[ix].to(device)).cpu().numpy()
        j = p.argmax(1)
        return np.array([0 if p[i, j[i]] <= 0 else j[i] + 1 for i in range(len(p))])

    ev_real = _realized(predict(ev_i), Msel[ev_i])
    va_real = _realized(predict(va_i), Msel[va_i])
    tr_real = _realized(predict(tr_i), Msel[tr_i])
    single = np.nan_to_num(Msel[ev_i, 0], nan=0.0)
    naive_or = np.array([max(0.0, np.nan_to_num(Msel[i], nan=-1e9).max()) for i in ev_i])
    elo, ehi = _boot(ev_real); slo, shi = _boot(single); olo, ohi = _boot(naive_or)

    mode = "frozen encoder + head" if args.freeze else "fine-tuned encoder"
    L = [f"# Dedicated-encoder router — {args.tag} ({args.encoder}, {mode})\n",
         f"Router = small encoder on the prompt TEXT -> swing-vector regression -> argmax. "
         f"{n} valid prompts ({len(tr_i)} train / {len(va_i)} val / {len(ev_i)} eval), K={K} moves, "
         f"same split as router_explore. Targets from swing_train.npz (m_swing); early stop on val MSE.\n",
         f"- **eval ΔRM {ev_real.mean():+.3f} [{elo:+.3f}, {ehi:+.3f}]**  (train {tr_real.mean():+.3f}, "
         f"val {va_real.mean():+.3f})",
         f"- single move {single.mean():+.3f} [{slo:+.3f}, {shi:+.3f}];  naive oracle {naive_or.mean():+.3f} "
         f"[{olo:+.3f}, {ohi:+.3f}]  (run's de-biased oracle ≈ +0.81)",
         "",
         "## Reading",
         "- **eval clears single (CI above the single point)** ⇒ a purpose-trained TEXT router extracts "
         "conditioning the LLM's own state could not ⇒ the method lives; scale + confirm on the honest test.",
         "- **eval ≈/below single AND train≈eval** ⇒ even a dedicated encoder can't predict the best move "
         "from the prompt text ⇒ strong evidence the per-prompt choice needs trial info, not just the text.",
         "- **train ≫ eval** ⇒ still overfitting; try --freeze, smaller encoder, or more data."]
    BASIS.mkdir(exist_ok=True)
    rpt = BASIS / f"s1_encoder_{args.tag}{'_frozen' if args.freeze else ''}_report.md"
    rpt.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {rpt}")


if __name__ == "__main__":
    main()
