#!/bin/bash
#SBATCH --job-name=aeS1-cond7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# S1.2 conditional controller at 7B (Qwen2.5-7B, steer_layer 18). Type-dependent positive control
# (A->hedge+, B->hedge-): does a conditional controller route different directions per type where
# the single-vector global cannot? Moving off 1.7B where steering is weak. Three arms (global/
# linear/mlp) in one job. ~2-3 GPU-h. If OOM in learn, drop pool.n_samples to 6 in the 7b config.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs

python src/steer_cond.py --phase all \
    --base-config configs/base_7b.yaml \
    --config configs/steer_cond_7b.yaml

echo "DONE — push results/steer_cond_7b/, basis/s1_cond_7b_report.md, results/cost_log.jsonl, slurm log."
