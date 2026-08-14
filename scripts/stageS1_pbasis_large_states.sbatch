#!/bin/bash
#SBATCH --job-name=aeS1-lgstate
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm-%j.out
# Cache read states for the scaled run (no generation, ~15-20 min) -> results/prompt_basis_large_7b/
# states.npz. Enables the offline router study (src/router_explore.py) to iterate architectures in
# seconds. Reuses the existing swing_train.npz + selection.json from the completed large run.

set -euo pipefail
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/prompt_basis.py --phase states \
    --base-config configs/base_7b.yaml \
    --config configs/prompt_basis_large_7b.yaml

echo "DONE — then on the login node: .venv/bin/python src/router_explore.py --tag large_7b"
echo "push results/prompt_basis_large_7b/states.npz + basis/s1_router_large_7b_report.md"
