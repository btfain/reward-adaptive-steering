#!/bin/bash
# One-time environment setup on the cluster (run from the repo root on a login
# node, or inside your first interactive session).
set -euo pipefail

# EDIT if your cluster uses environment modules for python:
# module load python/3.11

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Keep the HF cache off your home quota. EDIT $SCRATCH to your scratch path.
echo "export HF_HOME=${SCRATCH:-$HOME/scratch}/hf_cache" >> .venv/bin/activate
echo "setup done — submit scripts/stageA_compliance.sbatch first"
