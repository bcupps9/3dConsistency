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

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is not installed in this environment." >&2
  echo "Install it with: python -m pip install -U \"huggingface_hub[cli]\"" >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"

echo "MODEL_ROOT: ${MODEL_ROOT}"
echo "LVP_DIR:    ${LVP_DIR}"
echo "WAN21_DIR:  ${WAN21_DIR}"

# Enable Rust transfer backend for faster parallel downloads when available.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if [[ "${DOWNLOAD_LVP}" == "1" ]]; then
  echo ""
  echo "[1/2] Downloading LVP checkpoints..."
  mkdir -p "${LVP_DIR}"
  huggingface-cli download KempnerInstituteAI/LVP \
    --include "checkpoints/**" \
    --local-dir "${LVP_DIR}" \
    --local-dir-use-symlinks False

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
  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
    --local-dir "${WAN21_DIR}" \
    --local-dir-use-symlinks False
  echo "Wan2.1 ready at: ${WAN21_DIR}"
fi

echo ""
echo "Download complete."
