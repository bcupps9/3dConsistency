#!/usr/bin/env bash
set -euo pipefail

# Submit one layout job per incomplete slice (model/dataset/task) for a run.
#
# This speeds up backfill by fanning out work across many single-GPU jobs while
# preserving resume semantics (SKIP_EXISTING=1).
#
# Usage:
#   RUN_ID=20260218_batch1 bash scripts/submit_layout_missing.sh
#   bash scripts/submit_layout_missing.sh 20260218_batch1
#
# Optional env vars:
#   RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/<RUN_ID>
#   DATASETS=physics_iq,wisa80k
#   TARGETS=wan22,wan21,lvp
#   TASKS=t2v,i2v
#   MAX_SAMPLES=0
#   SKIP_EXISTING=1
#   CONTINUE_ON_ERROR=1
#   MISSING_CKPT_ACTION=skip
#   GPU_PARTITION=gpu_h200
#   GRES=gpu:nvidia_h200:1
#   CPUS_PER_TASK=8
#   MEMORY=200G
#   WALLTIME=24:00:00
#   LOCAL_SUBMIT=1
#   DRY_RUN=0

RUN_ID="${1:-${RUN_ID:-}}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>  (or set RUN_ID)" >&2
  exit 1
fi

SCRATCH_BASE="${SCRATCH_BASE:-/n/netscratch/ydu_lab/Lab/bcupps/results}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_BASE}/${RUN_ID}}"

DATASETS="${DATASETS:-physics_iq,wisa80k}"
TARGETS="${TARGETS:-wan22,wan21,lvp}"
TASKS="${TASKS:-t2v,i2v}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MISSING_CKPT_ACTION="${MISSING_CKPT_ACTION:-skip}"

GPU_PARTITION="${GPU_PARTITION:-gpu_h200}"
GRES="${GRES:-gpu:nvidia_h200:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-200G}"
WALLTIME="${WALLTIME:-24:00:00}"
LOCAL_SUBMIT="${LOCAL_SUBMIT:-1}"

DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT not found: ${RUN_ROOT}" >&2
  exit 1
fi

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a raw <<< "${csv}"
  out_ref=()
  local item trimmed
  for item in "${raw[@]}"; do
    trimmed="${item//[[:space:]]/}"
    [[ -z "${trimmed}" ]] && continue
    out_ref+=("${trimmed}")
  done
}

count_manifest_rows() {
  local manifest="$1"
  if [[ ! -f "${manifest}" ]]; then
    echo 0
    return
  fi
  grep -cve '^[[:space:]]*$' "${manifest}" || true
}

count_outputs() {
  local outputs_dir="$1"
  if [[ ! -d "${outputs_dir}" ]]; then
    echo 0
    return
  fi
  find "${outputs_dir}" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' '
}

csv_to_array "${DATASETS}" dataset_list
csv_to_array "${TARGETS}" target_list
csv_to_array "${TASKS}" task_list

if [[ "${#dataset_list[@]}" -eq 0 || "${#target_list[@]}" -eq 0 || "${#task_list[@]}" -eq 0 ]]; then
  echo "DATASETS/TARGETS/TASKS produced empty list." >&2
  exit 1
fi

echo "run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "datasets=${DATASETS}"
echo "targets=${TARGETS}"
echo "tasks=${TASKS}"
echo "resources: partition=${GPU_PARTITION} gres=${GRES} cpus=${CPUS_PER_TASK} mem=${MEMORY} walltime=${WALLTIME}"
echo

submitted=0
skipped=0

for model in "${target_list[@]}"; do
  for dataset in "${dataset_list[@]}"; do
    for task in "${task_list[@]}"; do
      manifest="${RUN_ROOT}/${model}/${dataset}/${task}/inputs/manifest.jsonl"
      outputs="${RUN_ROOT}/${model}/${dataset}/${task}/outputs"
      expected="$(count_manifest_rows "${manifest}")"
      got="$(count_outputs "${outputs}")"

      if [[ "${expected}" -eq 0 ]]; then
        echo "skip ${model}/${dataset}/${task}: manifest missing or empty"
        skipped=$((skipped + 1))
        continue
      fi

      if [[ "${got}" -ge "${expected}" ]]; then
        echo "skip ${model}/${dataset}/${task}: complete (${got}/${expected})"
        skipped=$((skipped + 1))
        continue
      fi

      echo "submit ${model}/${dataset}/${task}: incomplete (${got}/${expected})"
      cmd=(
        env
        "DATASET_NAMES=${dataset}"
        "TARGETS=${model}"
        "TASKS=${task}"
        "RUN_ROOT=${RUN_ROOT}"
        "GPU_PARTITION=${GPU_PARTITION}"
        "GRES=${GRES}"
        "CPUS_PER_TASK=${CPUS_PER_TASK}"
        "MEMORY=${MEMORY}"
        "WALLTIME=${WALLTIME}"
        "MAX_SAMPLES=${MAX_SAMPLES}"
        "SKIP_EXISTING=${SKIP_EXISTING}"
        "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
        "MISSING_CKPT_ACTION=${MISSING_CKPT_ACTION}"
        "LOCAL_SUBMIT=${LOCAL_SUBMIT}"
        "JOB_NAME=wm_${model}_${dataset}_${task}"
        bash
        scripts/submit_slurm_layout.sh
        "${RUN_ID}"
      )

      if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN: %q ' "${cmd[@]}"
        echo
      else
        "${cmd[@]}"
      fi
      submitted=$((submitted + 1))
    done
  done
done

echo
echo "Done. submitted=${submitted} skipped=${skipped} dry_run=${DRY_RUN}"
