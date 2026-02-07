#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
REMOTE_VENV="${REMOTE_VENV:-$PROJECT_DIR/.venv}"
PROJECT_RUN_DIR="${PROJECT_DIR}/runs/${RUN_ID}"

SCRATCH_BASE_DEFAULT="/n/netscratch/${USER}"
SCRATCH_BASE="${SCRATCH_BASE:-$SCRATCH_BASE_DEFAULT}"
SCRATCH_RUN_DIR="${SCRATCH_BASE}/3dConsistency/runs/${RUN_ID}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_RUN_DIR}"

RUN_DIR="${PROJECT_RUN_DIR}"
if [[ -d "${SCRATCH_BASE}" ]]; then
  mkdir -p "${SCRATCH_RUN_DIR}"
  RUN_DIR="${SCRATCH_RUN_DIR}"
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  # Example (replace with your cluster's modules):
  # module load cuda/12.1
  # module load python/3.10
fi

if [[ -f "${REMOTE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_VENV}/bin/activate"
else
  echo "Remote venv not found at ${REMOTE_VENV}. Create it and install requirements/remote.txt." >&2
  exit 1
fi

# TODO: Replace the command below with the actual Wan2.2 inference invocation.
# Example placeholder:
# python third_party/Wan2.2/infer.py --config configs/wan2.2.yaml --output "${RUN_DIR}"
echo "TODO: run Wan2.2 inference here, writing outputs to ${RUN_DIR}"

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
