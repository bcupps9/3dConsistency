#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a dedicated LVP environment with flash-attn diagnostics.
#
# Usage:
#   bash scripts/bootstrap_lvp_env.sh
#
# Optional overrides:
#   ENV_NAME=ei_world_model
#   PYTHON_VERSION=3.10
#   CUDA_MODULE=cuda/12.4.1-fasrc01
#   MINIFORGE_MODULE=Miniforge3/24.11.3-fasrc02
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   TORCH_VERSION=2.6.0
#   TORCHVISION_VERSION=0.21.0
#   INSTALL_FLASH_ATTN=1
#   STRICT_FLASH_ATTN=0
#   FLASH_ATTN_FORCE_SOURCE=1
#   FLASH_ATTN_VERSION=2.7.4.post1
#   PURGE_PIP_CACHE=1
#   VERBOSE_PIP=0
#   DIAG_DIR=~/projects/3dConsistency/.diag/lvp_manual

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
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-}"
PURGE_PIP_CACHE="${PURGE_PIP_CACHE:-1}"
VERBOSE_PIP="${VERBOSE_PIP:-0}"
DIAG_DIR="${DIAG_DIR:-$PROJECT_DIR/.diag/lvp_$(date +%Y%m%d_%H%M%S)}"
LVP_TENSORBOARD_COMPAT_VERSION="${LVP_TENSORBOARD_COMPAT_VERSION:-2.15.2}"

say() { echo ""; echo "==== $* ===="; }
warn() { echo "Warning: $*" >&2; }

mkdir -p "${DIAG_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "third_party/large-video-planner/requirements.txt" ]]; then
  echo "Missing third_party/large-video-planner/requirements.txt. Ensure submodule is initialized." >&2
  exit 1
fi

say "Node Snapshot"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname -f || hostname)"
  echo "glibc: $(ldd --version | head -n 1 2>/dev/null || echo unknown)"
} | tee "${DIAG_DIR}/node_info.txt"

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
hash -r

if [[ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
  echo "Expected CONDA_DEFAULT_ENV=${ENV_NAME}, got ${CONDA_DEFAULT_ENV:-unset}" >&2
  exit 1
fi

ENV_PY="${CONDA_PREFIX}/bin/python"
ACTUAL_PY="$(python -c 'import sys; print(sys.executable)')"
if [[ "${ACTUAL_PY}" != "${ENV_PY}" ]]; then
  echo "Expected python ${ENV_PY} but got ${ACTUAL_PY}" >&2
  exit 1
fi

say "Base Tooling"
python -m pip install -U pip setuptools wheel packaging ninja psutil
python -m pip -V | tee "${DIAG_DIR}/pip_identity.txt"

say "Install Torch"
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
python - <<'PY' | tee "${DIAG_DIR}/torch_info.txt"
import os
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("torch_file", torch.__file__)
print("torch_lib", os.path.join(os.path.dirname(torch.__file__), "lib"))
PY

TORCH_LIB_DIR="$("${ENV_PY}" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export TORCH_LIB_DIR
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

say "Install LVP Requirements (Without flash-attn)"
REQ_FILE="third_party/large-video-planner/requirements.txt"
TMP_REQ_FILE="$(mktemp "${TMPDIR:-/tmp}/lvp_requirements.XXXXXX.txt")"
grep -viE '^[[:space:]]*#?[[:space:]]*flash[_-]?attn([[:space:]]|=|$)' "${REQ_FILE}" > "${TMP_REQ_FILE}"

# TensorFlow 2.15 requires tensorboard < 2.16.
# If requirements contain tensorflow==2.15.x, force tensorboard to a compatible pin.
if grep -Eq '^[[:space:]]*tensorflow==2\.15(\.[0-9]+)?([[:space:]]*#.*)?$' "${TMP_REQ_FILE}"; then
  say "Patch tensorboard pin for tensorflow compatibility"
  awk -v tb="${LVP_TENSORBOARD_COMPAT_VERSION}" '
    {
      if ($0 ~ /^[[:space:]]*tensorboard==[0-9.]+([[:space:]]*#.*)?$/) {
        print "tensorboard==" tb
      } else {
        print $0
      }
    }
  ' "${TMP_REQ_FILE}" > "${TMP_REQ_FILE}.patched"
  mv "${TMP_REQ_FILE}.patched" "${TMP_REQ_FILE}"
fi

python -m pip install -r "${TMP_REQ_FILE}"
rm -f "${TMP_REQ_FILE}"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  say "Install flash-attn"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found. Cannot build flash-attn from source." >&2
    if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
      exit 1
    fi
    warn "continuing without flash-attn"
  else
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export MAX_JOBS
    export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}_lvp"
    mkdir -p "${TORCH_EXTENSIONS_DIR}"

    if [[ "${FLASH_ATTN_FORCE_SOURCE}" == "1" ]]; then
      export FLASH_ATTN_FORCE_BUILD=TRUE
    fi

    python -m pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
    if [[ "${PURGE_PIP_CACHE}" == "1" ]]; then
      python -m pip cache purge >/dev/null 2>&1 || true
      rm -rf ~/.cache/torch_extensions /tmp/torch_extensions_"${USER}"* 2>/dev/null || true
    else
      python -m pip cache remove flash-attn >/dev/null 2>&1 || true
    fi

    FLASH_PKG="flash-attn"
    if [[ -n "${FLASH_ATTN_VERSION}" ]]; then
      FLASH_PKG="flash-attn==${FLASH_ATTN_VERSION}"
    fi

    PIP_VERBOSE=()
    if [[ "${VERBOSE_PIP}" == "1" ]]; then
      PIP_VERBOSE=(-vvv)
    fi

    FLASH_CMD=(python -m pip install "${PIP_VERBOSE[@]}" --no-build-isolation --no-cache-dir)
    if [[ "${FLASH_ATTN_FORCE_SOURCE}" == "1" ]]; then
      FLASH_CMD+=(--no-binary :all:)
    fi
    FLASH_CMD+=("${FLASH_PKG}")

    if ! "${FLASH_CMD[@]}" 2>&1 | tee "${DIAG_DIR}/pip_flash_attn_install.log"; then
      if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
        echo "flash-attn installation failed and STRICT_FLASH_ATTN=1." >&2
        exit 1
      fi
      warn "flash-attn install failed; continuing without it"
    else
      if ! python -c "import flash_attn" >/dev/null 2>&1; then
        SO_PATH="$("${ENV_PY}" - <<'PY'
import glob, site
for d in site.getsitepackages():
    m = glob.glob(d + "/flash_attn_2_cuda*.so")
    if m:
        print(m[0])
        break
PY
)"
        if [[ -n "${SO_PATH}" && -f "${SO_PATH}" ]]; then
          ldd -v "${SO_PATH}" > "${DIAG_DIR}/ldd_flash_attn.txt" 2>&1 || true
          if grep -E 'GLIBC_2\.(29|3[0-9])' "${DIAG_DIR}/ldd_flash_attn.txt" >/dev/null 2>&1; then
            warn "flash-attn .so appears to require GLIBC newer than node baseline. See ${DIAG_DIR}/ldd_flash_attn.txt"
          fi
        fi
        if [[ "${STRICT_FLASH_ATTN}" == "1" ]]; then
          echo "flash-attn installed but import failed; STRICT_FLASH_ATTN=1." >&2
          exit 1
        fi
        warn "flash-attn installed but import failed; continuing without it"
      fi
    fi
  fi
fi

if [[ "${INSTALL_DOWNLOAD_TOOLS}" == "1" ]]; then
  say "Install Download Tools"
  python -m pip install -U "huggingface_hub[cli]" hf_transfer
fi

say "Done"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "LVP env ready: ${ENV_NAME}"
echo "Diagnostics: ${DIAG_DIR}"
