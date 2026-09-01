#!/bin/bash
#SBATCH --job-name=aeS1-candpl
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# P1(i)/(ii) REFERENCE matrix — 96 prompts x medoid+curated pool (220) x m_swing=2 ~= 43k gens / 4 shards
# ~= 8h each. Eval-only ground truth (NOT the pipeline): validates P1(ii) cost-aware on the REAL pool +
# auto(medoid)-vs-curated + feeds the banner. PREREQ: configs/candidates_medoid_combined_7b.txt committed.
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
