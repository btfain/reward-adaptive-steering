#!/bin/bash
#SBATCH --job-name=aeS1-sweep7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# S1.2 diagnostic sweep at 7B — characterize the steering ceiling and decide action-space-limit
# vs learning-limit for the real-RM null. Reuses results/steer_rm_7b/pool.jsonl (no regeneration).
# Per magnitude frac {0.05,0.10,0.15,0.20}: learned-global (train+heldout), contrastive, oracle
# (best-of-K in the reward-relevant subspace), + a fluency/diversity guard. ~3-5 GPU-h.
# If OOM: drop oracle.k_random or eval.oracle_samples; if slow: trim mag_fracs.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/steer_sweep.py \
    --base-config configs/base_7b.yaml \
    --config configs/steer_sweep_7b.yaml

echo "DONE — push basis/s1_sweep_7b_report.md, results/steer_sweep_7b/, results/cost_log.jsonl, slurm log."
