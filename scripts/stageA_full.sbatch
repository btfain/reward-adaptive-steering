#!/bin/bash
#SBATCH --job-name=aeA-full
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a5000|a6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-%j.out
# Duke CS cluster. Same notes as stageA_compliance.sbatch.
# Submit ONLY after the compliance gate is reviewed green (stage-gating rule).

set -euo pipefail
source .venv/bin/activate

python src/basis_extract.py --phase generate
python src/basis_extract.py --phase extract

echo "DONE — commit and push data/pairs/, basis/, results/ for review"
