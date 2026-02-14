#!/usr/bin/env bash
set -euo pipefail

# Environment Setup (LVP)
# Prerequisites:
# - Python 3.10
# - CUDA 12.1+ (for GPU support)
# - Conda or Mamba package manager
#
# Step 1: Create Conda Environment
#   conda create python=3.10 -n ei_world_model
#   conda activate ei_world_model
#
# Step 2: Install Dependencies
#   pip install -r requirements.txt
#   pip install flash-attn --no-build-isolation

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
ENV_NAME="${ENV_NAME:-ei_world_model}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

cd "${PROJECT_DIR}"

if [[ ! -f "third_party/large-video-planner/requirements.txt" ]]; then
  echo "Missing third_party/large-video-planner/requirements.txt. Ensure submodule is initialized." >&2
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

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel

cd "${PROJECT_DIR}/third_party/large-video-planner"
python -m pip install -r requirements.txt

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "LVP env ready: ${ENV_NAME}"
