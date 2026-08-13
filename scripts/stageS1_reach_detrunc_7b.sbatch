#!/bin/bash
#SBATCH --job-name=aeS1-rchdt7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# JOB B — steering reachability probe on the detruncated pool. RUN AFTER JOB A (reuses its pool at
# results/prompt_basis_detrunc_7b/pool.jsonl). Re-asks the +0.49 reachability finding when truncation
# is no longer the reward signal: does per-prompt fluent steering still move reward (the QUALITY axis)?
# Same prompts/RM as Job A => a valid steering-vs-prompting comparison. ~4-5 GPU-h.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/steer_reach.py \
    --base-config configs/base_7b.yaml \
    --config configs/steer_reach_detrunc_7b.yaml

echo "DONE (Job B) — push basis/s1_reach_detrunc_7b_report.md, results/steer_reach_detrunc_7b/, "
echo "results/cost_log.jsonl, slurm log."
