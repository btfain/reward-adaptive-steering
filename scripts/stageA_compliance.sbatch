#!/bin/bash
#SBATCH --job-name=aeA-compliance
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a5000|a6000|v100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
# Duke CS cluster (see cs.duke.edu/csl/faqs/slurm). Pre-flight check that these
# features exist on the partition:  sinfo -p compsci-gpu -o "%N %G %f"
# If submit fails on the constraint or the queue stalls, drop the constraint
# line — the code falls back to fp32 on non-Ampere GPUs (needs >= 16GB VRAM).
# Submit from the repo root: sbatch scripts/stageA_compliance.sbatch

set -euo pipefail
source .venv/bin/activate

# Smoke test doubles as the 1.7B + RM validation on this hardware.
python src/smoke_test.py
python src/basis_extract.py --phase prompts
python src/basis_extract.py --phase compliance

echo "DONE — commit and push data/compliance/ + results/ for review before submitting stageA_full"
