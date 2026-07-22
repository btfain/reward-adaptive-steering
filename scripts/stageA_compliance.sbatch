#!/bin/bash
#SBATCH --job-name=aeA-compliance
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
# Duke CS cluster. GPU type is pinned in the gres (gpu:a5000:1), NOT via
# --constraint: node features select nodes, and mixed p100+a5000 nodes can
# hand you the P100 (which modern torch kernels no longer support). If the
# a5000 queue stalls, gpu:a6000:1 is the fallback.
# Submit from the repo root: sbatch scripts/stageA_compliance.sbatch

set -euo pipefail
source .venv/bin/activate

# Smoke test doubles as the 1.7B + RM validation on this hardware.
python src/smoke_test.py
python src/basis_extract.py --phase prompts
python src/basis_extract.py --phase compliance

echo "DONE — commit and push data/compliance/ + results/ for review before submitting stageA_full"
