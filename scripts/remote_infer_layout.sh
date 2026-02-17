#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"
SCRATCH_BASE_DEFAULT="/n/netscratch/ydu_lab/Lab/bcupps/results"
SCRATCH_BASE="${SCRATCH_BASE:-$SCRATCH_BASE_DEFAULT}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_BASE}/${RUN_ID}}"

# Comma-separated list of datasets already prepared by prepare_inference_layout.py.
DATASET_NAMES="${DATASET_NAMES:-}"

# Comma-separated list of targets/tasks to run in sequence.
# Supported targets: wan22,wan21
# Supported tasks: t2v,i2v
TARGETS="${TARGETS:-wan22,wan21}"
TASKS="${TASKS:-t2v,i2v}"

# Optional controls.
MAX_SAMPLES="${MAX_SAMPLES:-0}"           # 0 means "all samples in manifest".
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"  # 1 means keep going if one sample fails.

MODEL_BASE="${MODEL_BASE:-/n/netscratch/ydu_lab/Lab/bcupps/models}"

WAN22_ENV="${WAN22_ENV:-${PROJECT_DIR}/.mamba/wan2}"
WAN22_T2V_CKPT_DIR="${WAN22_T2V_CKPT_DIR:-${MODEL_BASE}/Wan2.2-T2V-A14B}"
WAN22_I2V_CKPT_DIR="${WAN22_I2V_CKPT_DIR:-${MODEL_BASE}/Wan2.2-I2V-A14B}"
WAN22_SIZE_T2V="${WAN22_SIZE_T2V:-1280*720}"
WAN22_SIZE_I2V="${WAN22_SIZE_I2V:-1280*720}"

WAN21_ENV="${WAN21_ENV:-${PROJECT_DIR}/.mamba/wan21}"
WAN21_T2V_CKPT_DIR="${WAN21_T2V_CKPT_DIR:-${MODEL_BASE}/Wan2.1-T2V-14B}"
WAN21_I2V_CKPT_DIR="${WAN21_I2V_CKPT_DIR:-${MODEL_BASE}/Wan2.1-I2V-14B-480P}"
WAN21_SIZE_T2V="${WAN21_SIZE_T2V:-832*480}"
WAN21_SIZE_I2V="${WAN21_SIZE_I2V:-832*480}"

CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"

log() {
  echo
  echo "==== $* ===="
}

if [[ -z "${DATASET_NAMES}" ]]; then
  echo "DATASET_NAMES is required (comma-separated). Example: DATASET_NAMES=physics_iq,wisa80k" >&2
  exit 1
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR not found: ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT not found: ${RUN_ROOT}" >&2
  echo "Run prepare_inference_layout.py first." >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "${CUDA_MODULE}" || true
  module load "${MINIFORGE_MODULE}" || true
fi

if [[ ! -x "${CONDA_HOOK_BIN}" ]]; then
  echo "Conda hook binary not found at ${CONDA_HOOK_BIN}" >&2
  exit 1
fi

eval "$("${CONDA_HOOK_BIN}" shell.bash hook)"

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

activate_env_path() {
  local env_path="$1"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    echo "Expected env path not found: ${env_path}" >&2
    exit 1
  fi
  conda activate "${env_path}"
}

manifest_rows() {
  local manifest_path="$1"
  local max_samples="$2"
  python - "$manifest_path" "$max_samples" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
max_samples = int(sys.argv[2])

count = 0
with open(manifest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sample_id = str(row.get("sample_id", "")).replace("\t", " ")
        prompt = str(row.get("prompt", "")).replace("\t", " ").replace("\n", " ")
        image_path = str(row.get("image_path", "")).replace("\t", " ")
        output_video = str(row.get("output_video", "")).replace("\t", " ")
        print("\t".join([sample_id, prompt, image_path, output_video]))
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
PY
}

run_wan22_manifest() {
  local dataset_name="$1"
  local task="$2"
  local manifest_path="${RUN_ROOT}/wan22/${dataset_name}/${task}/inputs/manifest.jsonl"

  if [[ ! -f "${manifest_path}" ]]; then
    log "Skipping wan22/${dataset_name}/${task}: manifest missing (${manifest_path})"
    return 0
  fi

  local wan_task=""
  local size=""
  local ckpt_dir=""
  case "${task}" in
    t2v)
      wan_task="t2v-A14B"
      size="${WAN22_SIZE_T2V}"
      ckpt_dir="${WAN22_T2V_CKPT_DIR}"
      ;;
    i2v)
      wan_task="i2v-A14B"
      size="${WAN22_SIZE_I2V}"
      ckpt_dir="${WAN22_I2V_CKPT_DIR}"
      ;;
    *)
      echo "Unknown task '${task}' for wan22. Use t2v or i2v." >&2
      exit 1
      ;;
  esac

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "Wan2.2 checkpoint dir not found for ${task}: ${ckpt_dir}" >&2
    exit 1
  fi

  local sample_count=0
  while IFS=$'\t' read -r sample_id prompt image_path output_video; do
    [[ -z "${sample_id}" ]] && continue
    sample_count=$((sample_count + 1))

    local log_file="${RUN_ROOT}/wan22/${dataset_name}/${task}/logs/${sample_id}.log"
    mkdir -p "$(dirname "${output_video}")" "$(dirname "${log_file}")"

    local cmd=(
      python generate.py
      --task "${wan_task}"
      --size "${size}"
      --ckpt_dir "${ckpt_dir}"
      --offload_model True
      --convert_model_dtype
      --prompt "${prompt}"
      --save_file "${output_video}"
    )
    if [[ "${task}" == "i2v" ]]; then
      if [[ -z "${image_path}" || ! -f "${image_path}" ]]; then
        echo "wan22/${dataset_name}/${task}/${sample_id}: missing image_path (${image_path})" >&2
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi
      cmd+=(--image "${image_path}")
    fi

    {
      echo "[$(date -Is)] sample_id=${sample_id}"
      echo "[$(date -Is)] output=${output_video}"
      echo "[$(date -Is)] cmd=${cmd[*]}"
    } >"${log_file}"

    if ! "${cmd[@]}" >>"${log_file}" 2>&1; then
      echo "wan22/${dataset_name}/${task}/${sample_id} failed. See ${log_file}" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    fi
  done < <(manifest_rows "${manifest_path}" "${MAX_SAMPLES}")

  log "Finished wan22/${dataset_name}/${task} (${sample_count} samples)"
}

run_wan21_manifest() {
  local dataset_name="$1"
  local task="$2"
  local manifest_path="${RUN_ROOT}/wan21/${dataset_name}/${task}/inputs/manifest.jsonl"

  if [[ ! -f "${manifest_path}" ]]; then
    log "Skipping wan21/${dataset_name}/${task}: manifest missing (${manifest_path})"
    return 0
  fi

  local wan_task=""
  local size=""
  local ckpt_dir=""
  case "${task}" in
    t2v)
      wan_task="t2v-14B"
      size="${WAN21_SIZE_T2V}"
      ckpt_dir="${WAN21_T2V_CKPT_DIR}"
      ;;
    i2v)
      wan_task="i2v-14B"
      size="${WAN21_SIZE_I2V}"
      ckpt_dir="${WAN21_I2V_CKPT_DIR}"
      ;;
    *)
      echo "Unknown task '${task}' for wan21. Use t2v or i2v." >&2
      exit 1
      ;;
  esac

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "Wan2.1 checkpoint dir not found for ${task}: ${ckpt_dir}" >&2
    exit 1
  fi

  local sample_count=0
  while IFS=$'\t' read -r sample_id prompt image_path output_video; do
    [[ -z "${sample_id}" ]] && continue
    sample_count=$((sample_count + 1))

    local log_file="${RUN_ROOT}/wan21/${dataset_name}/${task}/logs/${sample_id}.log"
    mkdir -p "$(dirname "${output_video}")" "$(dirname "${log_file}")"

    local cmd=(
      python generate.py
      --task "${wan_task}"
      --size "${size}"
      --ckpt_dir "${ckpt_dir}"
      --offload_model True
      --t5_cpu
      --prompt "${prompt}"
      --save_file "${output_video}"
    )
    if [[ "${task}" == "i2v" ]]; then
      if [[ -z "${image_path}" || ! -f "${image_path}" ]]; then
        echo "wan21/${dataset_name}/${task}/${sample_id}: missing image_path (${image_path})" >&2
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi
      cmd+=(--image "${image_path}")
    fi

    {
      echo "[$(date -Is)] sample_id=${sample_id}"
      echo "[$(date -Is)] output=${output_video}"
      echo "[$(date -Is)] cmd=${cmd[*]}"
    } >"${log_file}"

    if ! "${cmd[@]}" >>"${log_file}" 2>&1; then
      echo "wan21/${dataset_name}/${task}/${sample_id} failed. See ${log_file}" >&2
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    fi
  done < <(manifest_rows "${manifest_path}" "${MAX_SAMPLES}")

  log "Finished wan21/${dataset_name}/${task} (${sample_count} samples)"
}

cd "${PROJECT_DIR}"

IFS=',' read -r -a target_list <<< "${TARGETS}"
IFS=',' read -r -a dataset_list <<< "${DATASET_NAMES}"
IFS=',' read -r -a task_list <<< "${TASKS}"

for raw_target in "${target_list[@]}"; do
  target="${raw_target//[[:space:]]/}"
  [[ -z "${target}" ]] && continue

  case "${target}" in
    wan22)
      log "Running Wan2.2 layout batch"
      activate_env_path "${WAN22_ENV}"
      cd "${PROJECT_DIR}/third_party/Wan2.2"
      for raw_dataset in "${dataset_list[@]}"; do
        dataset_name="${raw_dataset//[[:space:]]/}"
        [[ -z "${dataset_name}" ]] && continue
        for raw_task in "${task_list[@]}"; do
          task="${raw_task//[[:space:]]/}"
          [[ -z "${task}" ]] && continue
          run_wan22_manifest "${dataset_name}" "${task}"
        done
      done
      ;;
    wan21)
      log "Running Wan2.1 layout batch"
      activate_env_path "${WAN21_ENV}"
      cd "${PROJECT_DIR}/third_party/Wan2.1"
      for raw_dataset in "${dataset_list[@]}"; do
        dataset_name="${raw_dataset//[[:space:]]/}"
        [[ -z "${dataset_name}" ]] && continue
        for raw_task in "${task_list[@]}"; do
          task="${raw_task//[[:space:]]/}"
          [[ -z "${task}" ]] && continue
          run_wan21_manifest "${dataset_name}" "${task}"
        done
      done
      ;;
    "")
      ;;
    *)
      echo "Unknown target '${target}'. Supported layout targets: wan22,wan21" >&2
      exit 1
      ;;
  esac
done

log "Layout batch finished"
echo "run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "datasets=${DATASET_NAMES}"
echo "targets=${TARGETS}"
echo "tasks=${TASKS}"
