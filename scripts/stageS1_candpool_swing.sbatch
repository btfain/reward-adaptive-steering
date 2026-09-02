#!/bin/bash
#SBATCH --job-name=aeS1-cpswing
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# SWING-ONLY re-run — reuses the CACHED 4-shard base pool (pool_shard_*.jsonl already computed; base_ref
# stable). Now RESUMABLE + incrementally checkpointed: a wall-kill mid-swing loses <=5 prompts, and a
# resubmit continues from the last checkpoint. ~24 train prompts/shard x 220 cands x m_swing=2. time 14h;
# if a shard still walls, just RESUBMIT this script (it resumes). AFTER all 4 (login, offline):
#   .venv/bin/python src/prompt_basis.py --phase assemble --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
#   .venv/bin/python src/prompt_basis.py --phase select   --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
#   .venv/bin/python src/candidate_eval.py --tag candpool_7b
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/prompt_basis.py --phase swing --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
echo "swing shard ${SLURM_ARRAY_TASK_ID} done"
