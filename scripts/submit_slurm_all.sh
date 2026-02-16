#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bcupps}"
REMOTE_HOST="${REMOTE_HOST:-holylogin05}"
REMOTE_DIR="${REMOTE_DIR:-~/projects/3dConsistency}"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

# Comma-separated target list: wan22,wan21,lvp
TARGETS="${TARGETS:-wan22,wan21,lvp}"
RUN_MODE="${RUN_MODE:-smoke}"

GPU_PARTITION="${GPU_PARTITION:-gpu_test}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-128G}"
WALLTIME="${WALLTIME:-04:00:00}"
JOB_NAME="${JOB_NAME:-wm_all_infer}"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc 'set -euo pipefail
  cd ${REMOTE_DIR}
  mkdir -p runs/${RUN_ID}
  TARGETS=\"${TARGETS}\" RUN_MODE=\"${RUN_MODE}\" sbatch \
    --job-name=${JOB_NAME} \
    --partition=${GPU_PARTITION} \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --mem=${MEMORY} \
    --time=${WALLTIME} \
    --output=runs/${RUN_ID}/slurm-%j.out \
    scripts/remote_infer_all.sh ${RUN_ID}
'"

echo "Submitted run_id=${RUN_ID}"
echo "Targets=${TARGETS} Mode=${RUN_MODE}"
