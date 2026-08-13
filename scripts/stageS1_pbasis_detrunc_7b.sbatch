#!/bin/bash
#SBATCH --job-name=aeS1-pbdt7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# JOB A — detruncated, de-biased procedural-prompt basis + router at 7B. Generates its own base
# pool at max_new_tokens=256 (fixes the 128-cap truncation artifact), rediscovers the submodular
# basis on the clean reward, and runs the router with a val/test split for a de-biased conditioning
# ceiling. The pool it writes (results/prompt_basis_detrunc_7b/pool.jsonl) is REUSED by Job B.
# NO backprop through the LLM. ~7 GPU-h. If slow: cut pool.m_swing or candidate count.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase all \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_detrunc_7b.yaml

echo "DONE (Job A) — push basis/s1_pbasis_detrunc_7b_report.md, results/prompt_basis_detrunc_7b/, "
echo "results/cost_log.jsonl, slurm log. Then run Job B (stageS1_reach_detrunc_7b) which reuses the pool."
