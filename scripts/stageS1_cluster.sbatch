#!/bin/bash
#SBATCH --job-name=aeS1-cluster
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# P1(i) step 3/3 — LIGHT cluster the smoke-verified pool (reduce by --factor), save FULL clusters, and
# write three representations (medoid / random / LLM-summary) each unioned with curated. Summary needs the
# generator (GPU), ~1 gen/cluster (~15 min). PREREQ: smoke assemble wrote configs/candidates_verified_7b.txt.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/cluster_candidates.py --base-config configs/base_7b.yaml --k 200
echo "cluster done — full clusters + medoid/random/summary pools written; commit them."
