#!/bin/bash
#SBATCH --job-name=aeS1-candpl
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# P1(i)/(ii) REFERENCE matrix — SMALL prompt subsample (64) x LARGE pool (~120 candidates) x m_swing=2
# ~= 15k gens / 4 shards ~= 3h each. Eval-only ground truth (NOT the pipeline). Reuses prompt_basis.py
# sharded pool->swing. PREREQ: gencands wrote configs/candidates_combined_7b.txt AND it is committed/pulled.
# AFTER all 4 shards finish, on the login node (I/O + offline select, NO GPU):
#   .venv/bin/python src/prompt_basis.py --phase assemble --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
#   .venv/bin/python src/prompt_basis.py --phase select   --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
# Then P1(i) eval: src/candidate_eval.py (auto vs curated vs combined basis value).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/prompt_basis.py --phase shard --shard ${SLURM_ARRAY_TASK_ID}/4 \
    --base-config configs/base_7b.yaml --config configs/prompt_basis_candpool_7b.yaml
echo "candpool shard ${SLURM_ARRAY_TASK_ID} done"
