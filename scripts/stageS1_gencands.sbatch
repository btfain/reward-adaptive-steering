#!/bin/bash
#SBATCH --job-name=aeS1-gencand
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# P1(i) step 1/3 — generate a VERY LARGE raw candidate pool (~2k) + provenance (candidate->source prompts).
# Clustering happens AFTER the smoke test now. ~250 gen calls, ~25-40 min. AFTER: commit the raw files,
# then launch the sharded smoke test (stageS1_smoke).
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/gen_candidates.py --base-config configs/base_7b.yaml \
    --pool results/prompt_basis_large_7b/pool.jsonl --n_calls 250 \
    --out configs/candidates_raw_7b.txt --provenance configs/candidates_raw_7b.provenance.json
echo "gencands done — inspect configs/candidates_raw_7b.txt; commit raw + provenance."
