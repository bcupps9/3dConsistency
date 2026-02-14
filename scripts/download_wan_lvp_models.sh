#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_wan_lvp_models.sh
#   MODEL_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/models bash scripts/download_wan_lvp_models.sh
#
# Optional:
#   DOWNLOAD_LVP=0 DOWNLOAD_WAN21=1 bash scripts/download_wan_lvp_models.sh

MODEL_ROOT="${MODEL_ROOT:-$PWD}"
LVP_DIR="${LVP_DIR:-${MODEL_ROOT}/LVP}"
WAN21_DIR="${WAN21_DIR:-${MODEL_ROOT}/Wan2.1-I2V-14B-480P}"

DOWNLOAD_LVP="${DOWNLOAD_LVP:-1}"
DOWNLOAD_WAN21="${DOWNLOAD_WAN21:-1}"
AUTO_ACTIVATE_CONDA="${AUTO_ACTIVATE_CONDA:-1}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/projects/3dConsistency/.mamba/wan2}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"

if [[ "${AUTO_ACTIVATE_CONDA}" == "1" ]] && ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1 && [[ -x "${CONDA_HOOK_BIN}" ]]; then
    module load "${MINIFORGE_MODULE}" || true
    # shellcheck disable=SC1090
    eval "$("${CONDA_HOOK_BIN}" shell.bash hook)"
    conda activate "${CONDA_ENV_PATH}" || true
  fi
fi

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "No Hugging Face CLI found in this environment after env setup." >&2
  echo "Install it with: python -m pip install -U \"huggingface_hub[cli]\"" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"

echo "MODEL_ROOT: ${MODEL_ROOT}"
echo "LVP_DIR:    ${LVP_DIR}"
echo "WAN21_DIR:  ${WAN21_DIR}"

# Enable Rust transfer backend for faster downloads only when available.
if python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

if command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
else
  HF_CLI=(huggingface-cli download)
fi

if [[ "${DOWNLOAD_LVP}" == "1" ]]; then
  echo ""
  echo "[1/2] Downloading LVP checkpoints..."
  mkdir -p "${LVP_DIR}"
  "${HF_CLI[@]}" KempnerInstituteAI/LVP \
    --include "checkpoints/**" \
    --local-dir "${LVP_DIR}"

  if [[ -d "${LVP_DIR}/checkpoints" ]]; then
    mkdir -p "${LVP_DIR}/data"
    rm -rf "${LVP_DIR}/data/ckpts"
    mv "${LVP_DIR}/checkpoints" "${LVP_DIR}/data/ckpts"
  fi

  if [[ -f "${LVP_DIR}/data/ckpts/lvp_14B.ckpt" ]]; then
    echo "LVP ready at: ${LVP_DIR}/data/ckpts/lvp_14B.ckpt"
  else
    echo "Warning: expected LVP checkpoint not found at ${LVP_DIR}/data/ckpts/lvp_14B.ckpt" >&2
  fi
fi

if [[ "${DOWNLOAD_WAN21}" == "1" ]]; then
  echo ""
  echo "[2/2] Downloading Wan2.1 I2V 14B 480P..."
  mkdir -p "${WAN21_DIR}"
  "${HF_CLI[@]}" Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir "${WAN21_DIR}"
  echo "Wan2.1 ready at: ${WAN21_DIR}"
fi

echo ""
echo "Download complete."
