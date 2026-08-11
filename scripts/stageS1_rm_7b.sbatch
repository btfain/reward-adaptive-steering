#!/bin/bash
#SBATCH --job-name=aeS1-rm7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# S1.2 (real RM) at 7B — learn reward-driven steering (global/linear/mlp) against the given RM on
# UltraFeedback prompts. Core question: does LEARNED steering beat the A2 contrastive ~0 at 7B?
# On-policy held-out eval, paired vs base, BOTH RMs, vs best-of-n ceiling. Loads Qwen-7B + RM1 (+RM2
# in eval). ~3-5 GPU-h. If OOM in learn, drop pool.n_samples to 6 or max_new_tokens to 96.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/steer_rm.py --phase all \
    --base-config configs/base_7b.yaml \
    --config configs/steer_rm_7b.yaml

echo "DONE — push results/steer_rm_7b/, basis/s1_rm_7b_report.md, results/cost_log.jsonl, slurm log."
