#!/bin/bash
#SBATCH --job-name=aeNPOV-n0
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# S1-NPOV / N0 — cheap precondition gate for the (b)+(a) test. OLMo (frozen) revises biased WNC sentences
# under the NPOV guidebook; Qwen-7B is the held-out GOLD judge. Checks headroom, judge validity, variance.
# Phases are separate python calls so OLMo (gen) and Qwen (judge) never sit on the GPU together.
set -euo pipefail
source .venv/bin/activate                       # sets HF_HOME onto the big-quota scratch fs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# WNC (real data) lives on the big-quota fs next to the HF cache, NEVER home/git.
export NPOV_DATA="${HF_HOME:-$PWD/data}/wnc_data"
mkdir -p "$NPOV_DATA"
if [ ! -f "$NPOV_DATA/bias_data/WNC/biased.word.train" ]; then
  echo "fetching Wiki Neutrality Corpus into $NPOV_DATA ..."
  wget -q http://nlp.stanford.edu/projects/bias/bias_data.zip -P "$NPOV_DATA"
  (cd "$NPOV_DATA" && unzip -q bias_data.zip)
fi

python src/npov_n0.py --phase data
python src/npov_n0.py --phase gen   --base-config configs/base_olmo3sft.yaml
python src/npov_n0.py --phase judge
echo "NPOV N0 done"
