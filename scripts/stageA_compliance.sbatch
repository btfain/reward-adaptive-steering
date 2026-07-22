#!/bin/bash
#SBATCH --job-name=aeA-compliance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# EDIT for your cluster (then submit from the repo root: sbatch scripts/stageA_compliance.sbatch):
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail
source .venv/bin/activate

# Smoke test doubles as the 1.7B + RM validation on this hardware.
python src/smoke_test.py
python src/basis_extract.py --phase prompts
python src/basis_extract.py --phase compliance

echo "DONE — commit and push data/compliance/ + results/ for review before submitting stageA_full"
