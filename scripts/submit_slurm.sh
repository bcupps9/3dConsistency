#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="bcupps"
REMOTE_HOST="holylogin05"
REMOTE_DIR="~/projects/3dConsistency"

GPU_PARTITION="${GPU_PARTITION:-gpu}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc 'set -euo pipefail
  cd ${REMOTE_DIR}
  mkdir -p runs/${RUN_ID}
  sbatch \
    --job-name=wan2.2_infer \
    --partition=${GPU_PARTITION} \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=02:00:00 \
    --output=runs/${RUN_ID}/slurm-%j.out \
    scripts/remote_infer.sh ${RUN_ID}
'"

echo "Submitted run_id=${RUN_ID}"
