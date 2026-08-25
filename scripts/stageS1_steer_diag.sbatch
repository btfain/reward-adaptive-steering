#!/bin/bash
#SBATCH --job-name=aeS1-stdiag
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=results/slurm-%j.out
# Steering direction diagnostic — single GPU, ~256 gens (~15-20 min). Distinguishes harness-bug vs real
# centroid destruction, and sweeps magnitude; SAVES completion text to results/steer_dir_diag/diag.json.
# (A) own fitted δ should reproduce steer_reach ≈ +0.49 (else application bug -> steering_bakeoff invalid).
# (B) shared centroid swept over α (destructive at all α vs fluent-but-unhelpful at low α).
# AFTER it finishes, push for inspection:
#   git add results/steer_dir_diag/diag.json basis/s1_steer_dir_diag_report.md results/cost_log.jsonl
#   git commit -m "steering direction diagnostic results" && git push
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/steer_dir_diag.py --base-config configs/base_7b.yaml
echo "steer_dir_diag done"
