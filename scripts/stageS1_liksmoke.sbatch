#!/bin/bash
#SBATCH --job-name=aeS1-liksm
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# P1(i) LIKELIHOOD smoke (reward-free / interpretability) — does each move increase the model's PREFERENCE
# for the high-reward completion over the low-reward one (teacher-forced margin shift)? Sharded over the
# raw pool. Cheaper than reward smoke (forward-only, no generation, no RM). Runs in PARALLEL with the
# recluster — independent. AFTER all shards (login node): assemble -> compares to the reward filter:
#   .venv/bin/python src/likelihood_smoke.py --phase assemble
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/likelihood_smoke.py --phase test --shard ${SLURM_ARRAY_TASK_ID}/4 --base-config configs/base_7b.yaml
echo "likelihood smoke shard ${SLURM_ARRAY_TASK_ID} done"
