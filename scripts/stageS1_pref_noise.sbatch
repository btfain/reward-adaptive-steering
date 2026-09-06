#!/bin/bash
#SBATCH --job-name=aeRB-prefnoise
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=results/slurm-%j.out
# Step 1 of the RM-noise diagnostic: score UF-binarized preference pairs (chosen/rejected) under both RMs
# (no generation) to calibrate the RM noise scale sigma_eps against ground-truth labels. AFTER (login,
# offline):  .venv/bin/python src/rm_pref_noise.py --phase analyze
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/rm_pref_noise.py --phase score --base-config configs/base_7b.yaml --n 4000
echo "pref_noise score done"
