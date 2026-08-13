#!/bin/bash
# Environment setup on the Duke CS cluster. Builds an OS-INDEPENDENT env that also lives OFF the
# home quota:
#   * a pinned Miniconda python (default 3.11) on xtmp scratch  -> immune to system-python/OS upgrades
#     (e.g. Ubuntu 26.04) that break a venv riding /bin/python3;
#   * a venv created from that python, placed ON xtmp and symlinked as ./.venv, so every sbatch
#     `source .venv/bin/activate` keeps working while the multi-GB env stays off the home quota;
#   * the HF model cache on xtmp too.
# Idempotent — RE-RUN this any time the env breaks (e.g. after the nodes are upgraded). Override the
# python version with PYVER=3.12, or the scratch dir with SCRATCH_DIR=/path.
set -euo pipefail

# --- scratch (expanded xtmp preferred; keeps env + HF cache off the home quota) ---
if [ -z "${SCRATCH_DIR:-}" ]; then
    if mkdir -p "/usr/xtmp/$USER" 2>/dev/null; then SCRATCH_DIR="/usr/xtmp/$USER"; else SCRATCH_DIR="$HOME/scratch"; fi
fi
CONDA_DIR="$SCRATCH_DIR/miniforge3"
PYENV_DIR="$SCRATCH_DIR/ras_py"          # pinned-python conda env
VENV_DIR="$SCRATCH_DIR/ras_venv"         # the actual venv (off home quota)
PYVER="${PYVER:-3.11}"
echo "scratch: $SCRATCH_DIR   python: $PYVER"

# --- pinned, OS-independent python via Miniforge on scratch ---
# Miniforge defaults to conda-forge, so we avoid Anaconda's ToS-gated pkgs/main + pkgs/r channels
# (which otherwise fail with CondaToSNonInteractiveError on a non-interactive `conda create`).
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "installing Miniforge -> $CONDA_DIR"
    curl -sL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$CONDA_DIR"
    rm -f /tmp/mf.sh
fi
if [ ! -x "$PYENV_DIR/bin/python" ]; then
    "$CONDA_DIR/bin/conda" create -y -p "$PYENV_DIR" --override-channels -c conda-forge "python=$PYVER"
fi
PYTHON="$PYENV_DIR/bin/python"
echo "python: $("$PYTHON" --version)"

# --- venv on scratch, symlinked into the repo so `source .venv/bin/activate` still works ---
rm -rf "$VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
rm -rf .venv && ln -s "$VENV_DIR" .venv          # repo-local symlink -> scratch venv (gitignored)
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "pip install failed (often: no torch wheel for python $PYVER). Retry with e.g. PYVER=3.12,"
    echo "or relax the torch pin in requirements.txt, then re-run this script."; exit 1; }

# --- HF cache on scratch (off home quota; re-downloadable) ---
mkdir -p "$SCRATCH_DIR/hf_cache"
grep -q "HF_HOME=" "$VENV_DIR/bin/activate" || echo "export HF_HOME=$SCRATCH_DIR/hf_cache" >> "$VENV_DIR/bin/activate"

echo "setup done. env lives on scratch ($VENV_DIR), symlinked as ./.venv (off home quota)."
echo "verify on a compute node:  sbatch scripts/env_check.sbatch"
