#!/bin/bash
#SBATCH --job-name=aeS1-lgsh
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# SCALED run (600 prompts, 768 tok) — PARALLEL generation, 4 shard array tasks (~9h each). Each writes
# pool_shard_i.jsonl + swing_shard_i.npz to results/prompt_basis_large_7b/ (on xtmp, persists through
# the storage maintenance). Run the POST job (stageS1_pbasis_large_post.sbatch) AFTER all 4 finish —
# it is cheap-ish (~4-5h) and deadline-free, so it can wait until after tomorrow's xtmp downtime.
# PREREQ: data/prompts.json must contain the 'train_large' split (run src/expand_prompts.py first).

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase shard --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_large_7b.yaml

echo "DONE shard ${SLURM_ARRAY_TASK_ID}/4 — after all 4, run: sbatch scripts/stageS1_pbasis_large_post.sbatch"
