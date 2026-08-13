#!/bin/bash
#SBATCH --job-name=aeS1-envck
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=results/slurm-%j.out
# Post-maintenance env check: runs on a COMPUTE node and reports its python + whether the venv's
# imports work there. No `set -e` so it always finishes. If IMPORTS FAILED, the compute-node system
# python changed under the venv (the maintenance) => rebuild: bash scripts/cluster_setup.sh
echo "node:           $(hostname)"
echo "system python3: $(command -v python3) -> $(python3 --version 2>&1)"
source .venv/bin/activate 2>/dev/null || echo "venv activate FAILED"
echo "venv python:    $(command -v python) -> $(python --version 2>&1)"
python -c "import numpy, torch; print('IMPORTS OK  numpy', numpy.__version__, ' torch', torch.__version__)" \
    || echo "IMPORTS FAILED — compute-node python changed under the venv; rebuild: bash scripts/cluster_setup.sh"
