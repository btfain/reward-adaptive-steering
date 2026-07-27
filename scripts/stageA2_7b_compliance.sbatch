#!/bin/bash
#SBATCH --job-name=ae7b-compl
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
# 7B basis, step 1 of 2: pole-compliance check on Qwen2.5-7B (does it act out the
# poles when instructed?). Writes data/compliance_7b/samples.md — INSPECT that
# qualitatively before running the extract job. 1.7B artifacts are untouched (tag=7b).
# Prefetch first on the login node: .venv/bin/python scripts/prefetch.py

set -euo pipefail
source .venv/bin/activate

python src/basis_extract.py --phase compliance --config configs/base_7b.yaml

echo "DONE — push data/compliance_7b/ and review the sample sheet before extract."
