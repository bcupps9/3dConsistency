#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bcupps}"
REMOTE_HOST="${REMOTE_HOST:-login.rc.fas.harvard.edu}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/projects/3dConsistency}"
LOCAL_PROJECT_DIR="${LOCAL_PROJECT_DIR:-$HOME/projects/3dConsistency}"
LOCAL_SUBMIT="${LOCAL_SUBMIT:-auto}"  # auto|1|0

GPU_PARTITION="${GPU_PARTITION:-gpu}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-wan2.2_infer}"
GPUS="${GPUS:-1}"
GRES="${GRES:-gpu:${GPUS}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
WALLTIME="${WALLTIME:-02:00:00}"

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
    mkdir -p runs/${RUN_ID}
    sbatch \
      --job-name=${JOB_NAME} \
      --partition=${GPU_PARTITION} \
      --gres=${GRES} \
      --cpus-per-task=${CPUS_PER_TASK} \
      --mem=${MEMORY} \
      --time=${WALLTIME} \
      --output=runs/${RUN_ID}/slurm-%j.out \
      scripts/remote_infer.sh ${RUN_ID}
  "
else
  ssh "${REMOTE_USER}@${REMOTE_HOST}" "bash -lc 'set -euo pipefail
    cd ${REMOTE_PROJECT_DIR}
    mkdir -p runs/${RUN_ID}
    sbatch \
      --job-name=${JOB_NAME} \
      --partition=${GPU_PARTITION} \
      --gres=${GRES} \
      --cpus-per-task=${CPUS_PER_TASK} \
      --mem=${MEMORY} \
      --time=${WALLTIME} \
      --output=runs/${RUN_ID}/slurm-%j.out \
      scripts/remote_infer.sh ${RUN_ID}
  '"
fi

echo "Submitted run_id=${RUN_ID}"
