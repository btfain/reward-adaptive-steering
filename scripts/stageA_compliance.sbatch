#!/bin/bash
#SBATCH --job-name=aeA-compliance
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%j.out
# Duke CS cluster (see cs.duke.edu/csl/faqs/slurm). If the a5000/a6000 queue is
# long, drop the constraint line — the code falls back to fp32 on older GPUs.
# If you have a research-group partition, add: #SBATCH -A YOUR_GROUP -p YOUR_GROUP
# Submit from the repo root: sbatch scripts/stageA_compliance.sbatch

set -euo pipefail
source .venv/bin/activate

# Smoke test doubles as the 1.7B + RM validation on this hardware.
python src/smoke_test.py
python src/basis_extract.py --phase prompts
python src/basis_extract.py --phase compliance

echo "DONE — commit and push data/compliance/ + results/ for review before submitting stageA_full"
