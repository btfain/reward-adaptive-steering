#!/bin/bash
#SBATCH --job-name=aeMT-pair
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# M1b DENOISE — pairwise (relative) judging of move-a2 vs null-a2 over the CACHED generations (no regen).
# Lower-variance than absolute 1-5 and = the selection primitive; both orders cancel position bias.
# RESUMABLE + 4-way sharded. AFTER all 4 finish, assemble on the login node (no GPU):
#   .venv/bin/python src/mt_swing.py --phase pair_assemble --config configs/mt_swing_wildchat_v1.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/mt_swing.py --phase pairwise --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --config configs/mt_swing_wildchat_v1.yaml
echo "mt_pairwise shard ${SLURM_ARRAY_TASK_ID} done"
