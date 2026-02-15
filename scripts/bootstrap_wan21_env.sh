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
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
STRICT_FLASH_ATTN="${STRICT_FLASH_ATTN:-0}"
INSTALL_DOWNLOAD_TOOLS="${INSTALL_DOWNLOAD_TOOLS:-1}"
MAX_JOBS="${MAX_JOBS:-8}"

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
python -m pip install -U pip setuptools wheel packaging ninja psutil

# Install CUDA-compatible PyTorch before project deps.
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

REQ_FILE="third_party/Wan2.1/requirements.txt"
TMP_REQ_FILE="$(mktemp "${TMPDIR:-/tmp}/wan21_requirements.XXXXXX.txt")"
grep -viE '^[[:space:]]*flash[_-]?attn([[:space:]]|$)' "${REQ_FILE}" > "${TMP_REQ_FILE}"
REQ_FILE="${TMP_REQ_FILE}"

python -m pip install -r "${REQ_FILE}"

if [[ -f "${TMP_REQ_FILE}" ]]; then
  rm -f "${TMP_REQ_FILE}"
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi
  export MAX_JOBS
  if ! python -m pip install flash-attn --no-build-isolation; then
    if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
      echo "flash-attn installation failed and STRICT_FLASH_ATTN=1." >&2
      exit 1
    fi
    echo "Warning: flash-attn install failed; continuing without it (inference is still possible)." >&2
  fi
fi

if [[ "${INSTALL_DOWNLOAD_TOOLS}" == "1" ]]; then
  python -m pip install -U "huggingface_hub[cli]" hf_transfer
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Wan2.1 env ready: ${ENV_PATH}"
