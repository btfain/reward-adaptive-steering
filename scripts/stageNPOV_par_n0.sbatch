#!/bin/bash
#SBATCH --job-name=aeNPOV-parN0
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# S1-NPOV / N0 paragraph-level — OLMo (frozen) writes neutral encyclopedic paragraphs on real contested
# topics under the NPOV guidebook; Qwen-7B is the held-out GOLD judge, validated by polarity controls.
# Est ~1-1.5 GPU-hr; 6h cap (2x high-end, standing rule). Phases keep OLMo/Qwen off the GPU together.
set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# no external dataset -- topics live in the config
rm -f results/npov_par/prompts.json results/npov_par/gen_*.jsonl results/npov_par/npov_par_n0.npz

python src/npov_par_n0.py --phase gen   --base-config configs/base_olmo3sft.yaml
python src/npov_par_n0.py --phase judge
echo "NPOV paragraph N0 done"
