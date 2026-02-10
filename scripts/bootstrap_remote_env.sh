#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="${ENV_PATH:-$HOME/projects/3dConsistency/.mamba/wan2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

module load "${MINIFORGE_MODULE}"
eval "$(/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda shell.bash hook)"

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  mamba create -p "${ENV_PATH}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_PATH}"
python -m pip install -U pip setuptools wheel

# Install torch first to satisfy flash_attn build requirements later.
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

# Install remaining deps (may still fail on flash_attn on login nodes).
python -m pip install -r requirements/remote.txt

echo "Env ready at ${ENV_PATH}"
