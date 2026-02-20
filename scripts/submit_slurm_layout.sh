#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bcupps}"
REMOTE_HOST="${REMOTE_HOST:-login.rc.fas.harvard.edu}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/projects/3dConsistency}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$HOME/projects/3dConsistency}"
LOCAL_SUBMIT="${LOCAL_SUBMIT:-auto}"  # auto|1|0

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"

# Required: comma-separated dataset names prepared under RUN_ROOT.
DATASET_NAMES="${DATASET_NAMES:-}"

# Supported targets/tasks for layout runner.
TARGETS="${TARGETS:-wan22,wan21,lvp}"
TASKS="${TASKS:-t2v,i2v}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MISSING_CKPT_ACTION="${MISSING_CKPT_ACTION:-skip}"

GPU_PARTITION="${GPU_PARTITION:-gpu_test}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-128G}"
WALLTIME="${WALLTIME:-04:00:00}"
JOB_NAME="${JOB_NAME:-wm_layout_infer}"

SCRATCH_BASE_DEFAULT="/n/netscratch/ydu_lab/Lab/bcupps/results"
SCRATCH_BASE="${SCRATCH_BASE:-$SCRATCH_BASE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_BASE}/${RUN_ID}}"

# Cluster-specific default: gpu_h200 uses the nvidia_h200 GRES name.
if [[ -z "${GRES:-}" ]]; then
  if [[ "${GPU_PARTITION}" == "gpu_h200" ]]; then
    GRES="gpu:nvidia_h200:${GPUS}"
  else
    GRES="gpu:${GPUS}"
  fi
fi

if [[ -z "${DATASET_NAMES}" ]]; then
  echo "DATASET_NAMES is required. Example: DATASET_NAMES=physics_iq,wisa80k" >&2
  exit 1
fi

use_local_submit=0
case "${LOCAL_SUBMIT}" in
  1|true|yes)
    use_local_submit=1
    ;;
  0|false|no)
    use_local_submit=0
    ;;
  auto)
    if command -v sbatch >/dev/null 2>&1; then
      use_local_submit=1
    fi
    ;;
  *)
    echo "Invalid LOCAL_SUBMIT=${LOCAL_SUBMIT}. Use auto|1|0." >&2
    exit 1
    ;;
esac

if [[ "${use_local_submit}" == "1" ]]; then
  bash -lc "set -euo pipefail
    cd \"${LOCAL_PROJECT_DIR}\"
    mkdir -p \"${RUN_ROOT}\"
    DATASET_NAMES=\"${DATASET_NAMES}\" \
    TARGETS=\"${TARGETS}\" \
    TASKS=\"${TASKS}\" \
    MAX_SAMPLES=\"${MAX_SAMPLES}\" \
    CONTINUE_ON_ERROR=\"${CONTINUE_ON_ERROR}\" \
    SKIP_EXISTING=\"${SKIP_EXISTING}\" \
    MISSING_CKPT_ACTION=\"${MISSING_CKPT_ACTION}\" \
    RUN_ROOT=\"${RUN_ROOT}\" \
    sbatch \
      --job-name=${JOB_NAME} \
      --partition=${GPU_PARTITION} \
      --gres=${GRES} \
      --cpus-per-task=${CPUS_PER_TASK} \
      --mem=${MEMORY} \
      --time=${WALLTIME} \
      --output=${RUN_ROOT}/slurm-%j.out \
      scripts/remote_infer_layout.sh ${RUN_ID}
  "
else
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc 'set -euo pipefail
    cd ${REMOTE_PROJECT_DIR}
    mkdir -p ${RUN_ROOT}
    DATASET_NAMES=\"${DATASET_NAMES}\" \
    TARGETS=\"${TARGETS}\" \
    TASKS=\"${TASKS}\" \
    MAX_SAMPLES=\"${MAX_SAMPLES}\" \
    CONTINUE_ON_ERROR=\"${CONTINUE_ON_ERROR}\" \
    SKIP_EXISTING=\"${SKIP_EXISTING}\" \
    MISSING_CKPT_ACTION=\"${MISSING_CKPT_ACTION}\" \
    RUN_ROOT=\"${RUN_ROOT}\" \
    sbatch \
      --job-name=${JOB_NAME} \
      --partition=${GPU_PARTITION} \
      --gres=${GRES} \
      --cpus-per-task=${CPUS_PER_TASK} \
      --mem=${MEMORY} \
      --time=${WALLTIME} \
      --output=${RUN_ROOT}/slurm-%j.out \
      scripts/remote_infer_layout.sh ${RUN_ID}
  '"
fi

echo "Submitted layout run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "datasets=${DATASET_NAMES}"
echo "targets=${TARGETS} tasks=${TASKS}"
