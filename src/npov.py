"""Wikipedia NPOV domain (Subproject 1, the (b)+(a) test): Wiki Neutrality Corpus (WNC) loader, split, and
the neutralization task framing. The guidebook / rubric live in the yaml; the judge machinery is reused from
mt_judge (Qwen-7B as the held-out GOLD judge, OLMo as the frozen base we control).

Large data lives under $NPOV_DATA — a big-quota scratch filesystem — NEVER the home dir or git. Fetch once:
  wget http://nlp.stanford.edu/projects/bias/bias_data.zip -P $NPOV_DATA
  (cd $NPOV_DATA && unzip -q bias_data.zip)
WNC: Pryzant, Diehl Martinez, Dass, Kurohashi, Jurafsky, Yang, "Automatically Neutralizing Subjective Bias
in Text", AAAI 2020 — real biased->neutral Wikipedia edits (CC BY-SA, Wikipedia-derived).
"""

import os
import random
from pathlib import Path

from models import REPO_ROOT


def data_dir():
    """Big-quota scratch for WNC. $NPOV_DATA on the cluster (set next to HF_HOME); repo/data locally."""
    return Path(os.environ.get("NPOV_DATA") or (REPO_ROOT / "data" / "wnc"))


def _wnc_path(cfg):
    return data_dir() / cfg["data"]["wnc_subpath"]


def load_wnc(cfg):
    """Parse the WNC TSV into [{id, biased, neutral}], length-filtered. Columns (tab-separated):
    id, src_tok, tgt_tok, src_raw (biased), tgt_raw (neutral), [pos, rel...]. We read the raw cols 3,4."""
    p = _wnc_path(cfg)
    if not p.exists():
        raise FileNotFoundError(
            f"WNC not found at {p}. Fetch once onto the big-quota fs:\n"
            f"  wget http://nlp.stanford.edu/projects/bias/bias_data.zip -P {data_dir()}\n"
            f"  (cd {data_dir()} && unzip -q bias_data.zip)")
    d = cfg["data"]; lo, hi = d["min_chars"], d["max_chars"]
    out, seen = [], set()
    for ln in open(p, encoding="utf-8"):
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5:
            continue
        wid, biased, neutral = c[0], c[3].strip(), c[4].strip()
        if not biased or not neutral or biased == neutral:
            continue
        if not (lo <= len(biased) <= hi):
            continue
        if biased in seen:
            continue
        seen.add(biased)
        out.append({"id": wid, "biased": biased, "neutral": neutral})
    return out


def split(cfg):
    """Deterministic eval/discover split. eval -> N0 + N2 held-out; discover -> N1 move mining."""
    rows = load_wnc(cfg)
    random.Random(cfg["data"]["seed"]).shuffle(rows)
    ne, nd = cfg["data"]["n_eval"], cfg["data"]["n_discover"]
    return {"eval": rows[:ne], "discover": rows[ne:ne + nd]}


def neutralize_instruction(cfg, biased):
    """The user-turn task (the guidebook goes in the SYSTEM prompt, not here)."""
    return cfg["task_instruction"].format(passage=biased)
