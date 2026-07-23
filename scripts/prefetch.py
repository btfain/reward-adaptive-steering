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
synth = ROOT / "configs" / "synth.yaml"
if synth.exists():
    repos.append(yaml.safe_load(open(synth))["generator"]["model"])
for repo in repos:
    print(f"prefetching {repo}")
    snapshot_download(repo)
print("done")
