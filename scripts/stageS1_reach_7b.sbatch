#!/bin/bash
#SBATCH --job-name=aeS1-reach7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# S1.2 per-prompt free-δ reachability probe at 7B. For each of 200 prompts, learn an
# unconstrained full-rank δ by RWR on its pool at fluent caps {0.05,0.10}, evaluate on-policy.
# Headline: is there ANY fluent reward-increasing steering (fraction of best-of-n)? Plus a
# surrogate-optimization trace and a factorization diagnostic (effective rank + h->δ probe) for
# the joint-learning question. Reuses results/steer_rm_7b/pool.jsonl. RM1 only. ~4-6 GPU-h.
# If OOM: lower pool.n_samples (fewer completions per inner step). If slow: cut inner.steps.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/steer_reach.py \
    --base-config configs/base_7b.yaml \
    --config configs/steer_reach_7b.yaml

echo "DONE — push basis/s1_reach_7b_report.md, results/steer_reach_7b/, results/cost_log.jsonl, slurm log."
