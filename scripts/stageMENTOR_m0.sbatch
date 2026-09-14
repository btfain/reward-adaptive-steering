#!/bin/bash
#SBATCH --job-name=aeMENTOR-m0
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out
# MENTOR-BA / M0 — headroom + judge-validity read under the v2 mentoring guidebook (NO discovery/selection).
# OLMo (frozen) answers WildChat coding prompts under the guidebook; Qwen analytic gold judge scores them,
# plus appropriate-vs-inappropriate validity controls. Est ~45-60 min; 4h cap (2x high-end). Phases keep
# OLMo/Qwen off the GPU together.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

rm -f results/mentor_m0/prompts.json results/mentor_m0/gen_*.jsonl results/mentor_m0/mentor_m0.npz

python src/mentor_m0.py --phase gen   --base-config configs/base_olmo3sft.yaml
python src/mentor_m0.py --phase judge
echo "MENTOR M0 done"
