#!/bin/bash
#SBATCH --job-name=aeB1-train
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-1
#SBATCH --output=results/slurm-%A_%a.out
# B1 step 3/4 — online bandit training, ONE arm per array task in PARALLEL (task 0 = frozen head,
# task 1 = fine-tuned encoder), same roberta-base. CRASH-SAFE: per-epoch checkpoint (ckpt_<arm>.pt) +
# resume, so if a job hits the wall you just RESUBMIT this script and each arm continues from its last
# completed epoch (a finished arm re-runs nothing — its train_<arm>.json already exists and phase
# short-circuits). PYTHONUNBUFFERED=1 so epoch lines appear live in the slurm log. PREREQ: base_ref.json.
# AFTER both arms finish (train_frozen.json + train_finetune.json present) + assemble_prep, login node:
#     .venv/bin/python src/bandit_online.py --phase report --config configs/bandit_online_1500_7b.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
ARMS=(frozen finetune)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
if [ -f results/bandit_online_b1_1500/train_${ARM}.json ]; then
  echo "train arm $ARM already complete (train_${ARM}.json exists) — skipping"; exit 0
fi
python src/bandit_online.py --phase train --arm $ARM --config configs/bandit_online_1500_7b.yaml
echo "train arm $ARM done"
