#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
REMOTE_ENV="${REMOTE_ENV:-$PROJECT_DIR/.mamba/wan2}"
PROJECT_RUN_DIR="${PROJECT_DIR}/runs/${RUN_ID}"

SCRATCH_BASE_DEFAULT="/n/netscratch/ydu_lab/Lab/bcupps"
SCRATCH_BASE="${SCRATCH_BASE:-$SCRATCH_BASE_DEFAULT}"
SCRATCH_RUN_DIR="${SCRATCH_BASE}/results/${RUN_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_RUN_DIR}"

RUN_DIR="${PROJECT_RUN_DIR}"
if [[ -d "${SCRATCH_BASE}" ]]; then
  mkdir -p "${SCRATCH_RUN_DIR}"
  RUN_DIR="${SCRATCH_RUN_DIR}"
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.4.1-fasrc01
  module load Miniforge3/24.11.3-fasrc02
fi

if [[ -x "/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda" ]]; then
  eval "$(/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda shell.bash hook)"
else
  echo "Conda hook not found. Ensure Miniforge module is loaded." >&2
  exit 1
fi

if [[ -x "${REMOTE_ENV}/bin/python" ]]; then
  conda activate "${REMOTE_ENV}"
else
  echo "Remote env not found at ${REMOTE_ENV}. Run scripts/bootstrap_remote_env.sh first." >&2
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

MODEL_BASE="${MODEL_BASE:-/n/netscratch/ydu_lab/Lab/bcupps/models}"
MODEL_NAME="${MODEL_NAME:-Wan2.2-T2V-A14B}"
CKPT_DIR="${CKPT_DIR:-${MODEL_BASE}/${MODEL_NAME}}"
TASK="${TASK:-t2v-A14B}"
SIZE="${SIZE:-832*480}"
PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
SAVE_FILE="${SAVE_FILE:-${RUN_DIR}/t2v_A14B_${RUN_ID}.mp4}"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint dir not found: ${CKPT_DIR}" >&2
  echo "Set CKPT_DIR or MODEL_BASE/MODEL_NAME to where the weights live." >&2
  exit 1
fi

cd "${PROJECT_DIR}/third_party/Wan2.2"
python generate.py \
  --task "${TASK}" \
  --size "${SIZE}" \
  --ckpt_dir "${CKPT_DIR}" \
  --offload_model True \
  --convert_model_dtype \
  --prompt "${PROMPT}" \
  --save_file "${SAVE_FILE}"

if [[ "${RUN_DIR}" != "${PROJECT_RUN_DIR}" ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -az "${RUN_DIR}/" "${PROJECT_RUN_DIR}/"
  else
    cp -R "${RUN_DIR}/" "${PROJECT_RUN_DIR}/"
  fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
  if ls "${PROJECT_RUN_DIR}"/*.png >/dev/null 2>&1; then
    ffmpeg -y -framerate 24 -pattern_type glob -i "${PROJECT_RUN_DIR}/*.png" -c:v libx264 -pix_fmt yuv420p "${PROJECT_RUN_DIR}/output.mp4"
  fi
fi
