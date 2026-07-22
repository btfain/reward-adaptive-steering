"""Pre-download model weights into the HF cache without loading them into RAM.

Run on the cluster LOGIN node (which has reliable internet) if compute-node
downloads ever fail:  .venv/bin/python scripts/prefetch.py
"""

from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs" / "base.yaml"))
for repo in (cfg["base_model"], cfg["reward_model"]):
    print(f"prefetching {repo}")
    snapshot_download(repo)
print("done")
