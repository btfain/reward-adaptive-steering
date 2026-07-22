#!/bin/bash
# One-time environment setup on the Duke CS cluster. Run from the repo root on
# a login node. No module system needed per the cluster docs; falls back to
# Miniconda if the system python3 is too old for torch (needs >= 3.10).
set -euo pipefail

SCRATCH_DIR="${SCRATCH_DIR:-/usr/xtmp/$USER}"   # EDIT if your scratch lives elsewhere

PYV=$(python3 -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])' || echo 0)
if [ "$PYV" -lt 310 ]; then
    echo "system python3 too old ($PYV) — installing Miniconda into $SCRATCH_DIR/miniconda3"
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$SCRATCH_DIR/miniconda3"
    PYTHON="$SCRATCH_DIR/miniconda3/bin/python3"
else
    PYTHON=python3
fi

"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Keep the ~5GB HF model cache off the home quota (cluster is not backed up;
# everything in the cache is re-downloadable anyway).
mkdir -p "$SCRATCH_DIR/hf_cache"
echo "export HF_HOME=$SCRATCH_DIR/hf_cache" >> .venv/bin/activate
echo "setup done — submit scripts/stageA_compliance.sbatch first"
