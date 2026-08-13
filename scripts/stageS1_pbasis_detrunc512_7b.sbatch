#!/bin/bash
#SBATCH --job-name=aeS1-pb512
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=results/slurm-%j.out
# LARGE-A (validity): detruncated prompt-basis at max_new_tokens=512, prompts held at 120/50 for a
# clean one-variable comparison to the 256 run. Regenerates its own 512-token base pool (reused by a
# later larger B). NO LLM backprop. ~7-9 GPU-h (many gens finish before 512, so <2x the 256 run).
# NOTE: --time=14:00:00 assumes the partition allows >8h; if it caps at 8h, lower max_new_tokens to 384.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase all \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_detrunc512_7b.yaml

echo "DONE (large-A 512) — push basis/s1_pbasis_detrunc512_7b_report.md, results/prompt_basis_detrunc512_7b/, results/cost_log.jsonl, slurm log."
