#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="bcupps"
REMOTE_HOST="holylogin05"
REMOTE_DIR="~/projects/3dConsistency"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

mkdir -p "runs/${RUN_ID}"
rsync -az --progress \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/runs/${RUN_ID}/" "runs/${RUN_ID}/"

if command -v ffmpeg >/dev/null 2>&1; then
  if [[ ! -f "runs/${RUN_ID}/output.mp4" ]] && ls "runs/${RUN_ID}"/*.png >/dev/null 2>&1; then
    ffmpeg -y -framerate 24 -pattern_type glob -i "runs/${RUN_ID}/*.png" -c:v libx264 -pix_fmt yuv420p "runs/${RUN_ID}/output.mp4"
  fi
fi
