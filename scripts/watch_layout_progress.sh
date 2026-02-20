#!/usr/bin/env bash
set -euo pipefail

# Live monitor for layout inference runs.
#
# Usage:
#   bash scripts/watch_layout_progress.sh --run-id 20260218_batch1
#   RUN_ROOT=/n/netscratch/ydu_lab/Lab/bcupps/results/20260218_batch1 \
#     bash scripts/watch_layout_progress.sh
#
# Optional env vars:
#   INTERVAL=60               # Refresh cadence in seconds.
#   TRACK_USER=$USER          # Slurm user to inspect.
#   JOB_NAME=wm_layout_infer  # Slurm job name filter.

RUN_ID="${RUN_ID:-}"
RUN_ROOT="${RUN_ROOT:-}"
INTERVAL="${INTERVAL:-60}"
TRACK_USER="${TRACK_USER:-${USER:-}}"
JOB_NAME="${JOB_NAME:-wm_layout_infer}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:-}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ROOT}" ]]; then
  if [[ -z "${RUN_ID}" ]]; then
    echo "Provide --run-id <id> or --run-root <path> (or set RUN_ID/RUN_ROOT)." >&2
    exit 1
  fi
  RUN_ROOT="/n/netscratch/ydu_lab/Lab/bcupps/results/${RUN_ID}"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(basename "${RUN_ROOT}")"
fi

if ! [[ "${INTERVAL}" =~ ^[0-9]+$ ]] || [[ "${INTERVAL}" -lt 1 ]]; then
  echo "INTERVAL must be a positive integer. Got: ${INTERVAL}" >&2
  exit 1
fi

print_lvp_counts() {
  local lvp_root="$1"
  if [[ ! -d "${lvp_root}" ]]; then
    echo "LVP root missing: ${lvp_root}"
    return
  fi

  local found=0
  local dataset_dir task_dir manifest outputs expected got
  for dataset_dir in "${lvp_root}"/*; do
    [[ -d "${dataset_dir}" ]] || continue
    for task_dir in "${dataset_dir}"/*; do
      [[ -d "${task_dir}" ]] || continue
      found=1
      manifest="${task_dir}/inputs/manifest.jsonl"
      outputs="${task_dir}/outputs"
      expected=0
      got=0
      if [[ -f "${manifest}" ]]; then
        expected="$(grep -cve '^[[:space:]]*$' "${manifest}" || true)"
      fi
      if [[ -d "${outputs}" ]]; then
        got="$(find "${outputs}" -maxdepth 1 -type f -name '*.mp4' -size +0c 2>/dev/null | wc -l | tr -d ' ')"
      fi
      printf "lvp/%s/%s: outputs=%s/%s\n" \
        "$(basename "${dataset_dir}")" \
        "$(basename "${task_dir}")" \
        "${got}" "${expected}"
    done
  done

  if [[ "${found}" -eq 0 ]]; then
    echo "No LVP dataset/task layout dirs found under ${lvp_root}"
  fi
}

print_latest_lvp_log_snippets() {
  local run_root="$1"
  local log
  mapfile -t logs < <(find "${run_root}/lvp" -type f -name 'lvp_run.log' 2>/dev/null | sort)
  if [[ "${#logs[@]}" -eq 0 ]]; then
    echo "No lvp_run.log files yet."
    return
  fi

  for log in "${logs[@]}"; do
    echo "--- ${log}"
    grep -nE "Error executing job|Traceback|RuntimeError|ValueError|FAILED|Sampling:|Building model|Executing task|Outputs will be saved to" "${log}" | tail -n 8 || true
  done
}

print_recent_slurm_state() {
  local rr="$1"
  local start_date
  start_date="$(date -d '3 days ago' +%F 2>/dev/null || date +%F)"

  echo "Active queue (${TRACK_USER}, name=${JOB_NAME}):"
  squeue -u "${TRACK_USER}" -n "${JOB_NAME}" -o "%.18i %.9T %.10M %.30R" 2>/dev/null || true
  echo

  echo "Recent accounting rows touching this run_root:"
  sacct -S "${start_date}" -u "${TRACK_USER}" --name="${JOB_NAME}" \
    --format=JobIDRaw,State,ExitCode,Start,Elapsed,StdOut%140 -P 2>/dev/null \
    | awk -F'|' -v rr="${rr}" '
        NR==1 {print; next}
        $6 ~ rr {print}
      ' \
    | tail -n 12
}

while true; do
  clear || true
  echo "== Layout Tracker =="
  echo "time:    $(date -Is)"
  echo "run_id:  ${RUN_ID}"
  echo "run_root:${RUN_ROOT}"
  echo

  print_recent_slurm_state "${RUN_ROOT}"
  echo

  echo "LVP output progress:"
  print_lvp_counts "${RUN_ROOT}/lvp"
  echo

  latest_progress="$(ls -1t "${RUN_ROOT}"/progress_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "${latest_progress}" ]]; then
    echo "Latest progress log: ${latest_progress}"
    tail -n 20 "${latest_progress}" || true
  else
    echo "No progress_*.log found yet."
  fi
  echo

  echo "Latest LVP log snippets:"
  print_latest_lvp_log_snippets "${RUN_ROOT}"
  echo
  echo "Refresh in ${INTERVAL}s. Ctrl-C to stop."
  sleep "${INTERVAL}"
done

