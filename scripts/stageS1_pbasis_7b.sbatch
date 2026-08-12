#!/bin/bash
#SBATCH --job-name=aeS1-pbasis7b
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out
# S1 (revised) procedural-prompt basis + router at 7B. swing: RM-score each (train prompt x
# candidate move); select: greedy submodular basis (value-vs-K); route: layer-swept classifier,
# held-out ΔRM vs best-single-move / oracle-over-basis. Reuses results/steer_rm_7b/pool.jsonl for
# base(x). RM1 only, NO backprop through the LLM (lean). ~4-5 GPU-h (swing precompute dominates).
# If slow: cut pool.m_swing or the candidate file; if OOM (unlikely, inference-only): nothing to tune.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase all \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_7b.yaml

echo "DONE — push basis/s1_pbasis_7b_report.md, results/prompt_basis_7b/, results/cost_log.jsonl, slurm log."
