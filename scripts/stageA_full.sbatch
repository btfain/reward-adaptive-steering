#!/bin/bash
#SBATCH --job-name=aeA-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm-%j.out
# EDIT for your cluster:
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT
#
# Submit ONLY after the compliance gate is reviewed green (stage-gating rule).

set -euo pipefail
source .venv/bin/activate

python src/basis_extract.py --phase generate
python src/basis_extract.py --phase extract

echo "DONE — commit and push data/pairs/, basis/, results/ for review"
