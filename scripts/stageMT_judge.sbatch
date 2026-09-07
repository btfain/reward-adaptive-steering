#!/bin/bash
#SBATCH --job-name=aeMT-judge
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# M1b judge, RESUMABLE + 4-way SHARDED (gen is already cached in results/mt_wildchat_v1/gen.jsonl).
# Each array task scores its shard into scores_shard_<i>.jsonl; re-runnable (skips done). AFTER all 4
# finish, assemble on the login node (no GPU):
#   .venv/bin/python src/mt_swing.py --phase assemble --config configs/mt_swing_wildchat_v1.yaml
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/mt_swing.py --phase judge --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --config configs/mt_swing_wildchat_v1.yaml
echo "mt_judge shard ${SLURM_ARRAY_TASK_ID} done"
