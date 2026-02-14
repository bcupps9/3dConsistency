#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a dedicated Wan2.1 environment on cluster/local.
#
# Usage:
#   bash scripts/bootstrap_wan21_env.sh
#
# Optional overrides:
#   ENV_PATH=~/projects/3dConsistency/.mamba/wan21
#   PYTHON_VERSION=3.10
#   CUDA_MODULE=cuda/12.4.1-fasrc01
#   MINIFORGE_MODULE=Miniforge3/24.11.3-fasrc02
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
ENV_PATH="${ENV_PATH:-$PROJECT_DIR/.mamba/wan21}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

cd "${PROJECT_DIR}"

if [[ ! -f "third_party/Wan2.1/requirements.txt" ]]; then
  echo "Missing third_party/Wan2.1/requirements.txt. Ensure submodule is initialized." >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "${CUDA_MODULE}" || true
  module load "${MINIFORGE_MODULE}"
fi

if [[ ! -x "${CONDA_HOOK_BIN}" ]]; then
  echo "Conda hook binary not found at ${CONDA_HOOK_BIN}" >&2
  exit 1
fi

eval "$("${CONDA_HOOK_BIN}" shell.bash hook)"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  mamba create -p "${ENV_PATH}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_PATH}"
python -m pip install -U pip setuptools wheel

# Install CUDA-compatible PyTorch before project deps.
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
python -m pip install -r third_party/Wan2.1/requirements.txt

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Wan2.1 env ready: ${ENV_PATH}"
