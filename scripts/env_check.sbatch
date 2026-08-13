#!/bin/bash
#SBATCH --job-name=aeS1-envck
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=results/slurm-%j.out
# Post-maintenance env check on a COMPUTE node: python + imports + CUDA + bf16 (the GPU-driver upgrade
# means "imports OK" is not enough — the jobs need working CUDA). No `set -e` so it always finishes.
# If anything FAILS: rebuild the venv against 26.04's python + new drivers => bash scripts/cluster_setup.sh
echo "node:           $(hostname)"
echo "system python3: $(command -v python3) -> $(python3 --version 2>&1)"
source .venv/bin/activate 2>/dev/null || echo "venv activate FAILED"
python - <<'PY'
import sys; print("venv python:   ", sys.version.split()[0])
try:
    import numpy; print("numpy:         ", numpy.__version__)
except Exception as e: print("numpy FAIL:    ", repr(e))
try:
    import torch
    print("torch:         ", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:           ", torch.cuda.get_device_name(0))
        x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
        print("bf16 matmul OK:", round((x @ x).float().sum().item(), 1))
    else:
        print("=> CUDA NOT available: driver/runtime mismatch after the upgrade; rebuild the venv.")
except Exception as e:
    import traceback; print("torch FAIL:    ", repr(e)); traceback.print_exc()
PY
echo "If any line FAILED or CUDA is unavailable -> bash scripts/cluster_setup.sh (rebuild), then re-run this."
