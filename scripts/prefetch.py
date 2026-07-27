"""Pre-download model weights into the HF cache without loading them into RAM.

Run on the cluster LOGIN node (which has reliable internet) if compute-node
downloads ever fail:  .venv/bin/python scripts/prefetch.py
"""

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
repos = [cfg["base_model"], cfg["reward_model"]]
if cfg.get("reward_model_alt"):
    repos.append(cfg["reward_model_alt"])
synth = ROOT / "configs" / "synth.yaml"
if synth.exists():
    repos.append(yaml.safe_load(open(synth))["generator"]["model"])
b7 = ROOT / "configs" / "base_7b.yaml"
if b7.exists():
    repos.append(yaml.safe_load(open(b7))["base_model"])
for repo in dict.fromkeys(repos):
    print(f"prefetching {repo}")
    snapshot_download(repo)
print("done")
