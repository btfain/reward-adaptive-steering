#!/bin/bash
#SBATCH --job-name=aeS1-gencand
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# P1(i) step 1/2 — generate a LARGE auto candidate pool (~100) from contrastive reward signal, then union
# with the curated pool -> configs/candidates_combined_7b.txt (auto rows first). ~30 gen calls, ~20-30 min.
# Large pool is deliberate: top-k + router + cost-aware selection let (i)/(ii) exploit many candidates.
# AFTER it finishes, commit the candidate files + launch the swing-matrix job (stageS1_candpool_shard).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/gen_candidates.py --base-config configs/base_7b.yaml \
    --pool results/prompt_basis_large_7b/pool.jsonl \
    --n_calls 30 --target 100 \
    --out configs/candidates_auto_7b.txt --combined_out configs/candidates_combined_7b.txt
echo "gencands done — inspect configs/candidates_auto_7b.txt + candidates_combined_7b.txt"
