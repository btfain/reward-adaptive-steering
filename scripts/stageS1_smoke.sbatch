#!/bin/bash
#SBATCH --job-name=aeS1-smoke
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=results/slurm-%A_%a.out
# P1(i) step 2/3 — smoke-test the raw pool IN-CONTEXT (on each candidate's source prompts vs cached base
# reward), sharded over candidates. Cull wrong guesses (in-context swing<=0). FLOOR filter, NOT top-K.
# ~1000 candidates/4 shards x 3 source x 2 samples ~= 1.5k gens/shard (~1-1.5h). PREREQ: raw files committed.
# AFTER all shards finish (login node, offline): assemble -> verified pool + provenance:
#   .venv/bin/python src/smoke_candidates.py --phase assemble
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/smoke_candidates.py --phase test --shard ${SLURM_ARRAY_TASK_ID}/4 --base-config configs/base_7b.yaml
echo "smoke test shard ${SLURM_ARRAY_TASK_ID} done"
