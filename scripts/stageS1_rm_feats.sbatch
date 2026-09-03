#!/bin/bash
#SBATCH --job-name=aeS1-rmfeat
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=results/slurm-%j.out
# Capture diagnostic — is the routing info-limit representational or fundamental?
#  1) RM(prompt) features            (RM-only forward, fast)
#  2) base gens WITH TEXT for the 96 train prompts   (7B gen, ~18 min: 96 x 4)
#  3) RM(prompt, base) features      (RM forward over prompt+median-RM base gen)
#  4) capture_diag report            (generic vs rm_prompt vs rm_prompt_base, best-of-2 headroom captured)
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python src/extract_rm_feats.py --which prompt --base-config configs/base_7b.yaml --n 96
python src/gen_base_text.py --base-config configs/base_7b.yaml --n 96 --n_base 4
python src/extract_rm_feats.py --which prompt_base --base-config configs/base_7b.yaml --n 96 \
    --base-jsonl results/end_banner_7b/base_text_train.jsonl
python src/capture_diag.py --K 8 --seeds 12
echo "rm_feats + capture_diag done"
