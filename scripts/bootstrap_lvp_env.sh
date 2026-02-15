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
STRICT_FLASH_ATTN="${STRICT_FLASH_ATTN:-0}"
INSTALL_DOWNLOAD_TOOLS="${INSTALL_DOWNLOAD_TOOLS:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
MAX_JOBS="${MAX_JOBS:-8}"
FLASH_ATTN_FORCE_SOURCE="${FLASH_ATTN_FORCE_SOURCE:-1}"

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
python -m pip install -U pip setuptools wheel packaging ninja psutil

cd "${PROJECT_DIR}/third_party/large-video-planner"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements.txt

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi
  export MAX_JOBS
  python -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
  FLASH_CMD=(python -m pip install flash-attn --no-build-isolation)
  if [[ "${FLASH_ATTN_FORCE_SOURCE}" == "1" ]]; then
    FLASH_CMD+=(--no-binary flash-attn)
  fi
  if ! "${FLASH_CMD[@]}"; then
    if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
      echo "flash-attn installation failed and STRICT_FLASH_ATTN=1." >&2
      exit 1
    fi
    echo "Warning: flash-attn install failed; continuing without it (inference is still possible)." >&2
  elif ! python -c "import flash_attn" >/dev/null 2>&1; then
    if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
      echo "flash-attn installed but import failed; STRICT_FLASH_ATTN=1." >&2
      exit 1
    fi
    echo "Warning: flash-attn installed but cannot be imported; continuing without it." >&2
  fi
fi

if [[ "${INSTALL_DOWNLOAD_TOOLS}" == "1" ]]; then
  python -m pip install -U "huggingface_hub[cli]" hf_transfer
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "LVP env ready: ${ENV_NAME}"
