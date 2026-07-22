#!/bin/bash
#SBATCH --job-name=aeA-chalfix
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# Re-extract the basis with the challenge pair filter active, then re-run the
# sanity sweep for challenge only (other axes' sweep files are kept).

set -euo pipefail
source .venv/bin/activate

python src/basis_extract.py --phase extract
python src/steer_sanity.py --redo challenge_accommodate

echo "DONE — commit and push basis/, results/"
