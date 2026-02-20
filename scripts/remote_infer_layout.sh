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
# Supported targets: wan22,wan21,lvp
# Supported tasks: t2v,i2v
TARGETS="${TARGETS:-wan22,wan21,lvp}"
TASKS="${TASKS:-t2v,i2v}"

# Optional controls.
MAX_SAMPLES="${MAX_SAMPLES:-0}"           # 0 means "all samples in manifest".
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"  # 1 means keep going if one sample fails.
SKIP_EXISTING="${SKIP_EXISTING:-1}"       # 1 means skip samples with existing non-empty output_video.
HEARTBEAT_SECS="${HEARTBEAT_SECS:-120}"   # Periodic progress ping while a sample is running.
MISSING_CKPT_ACTION="${MISSING_CKPT_ACTION:-skip}"  # skip|fail when required checkpoint dir is unavailable.

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

LVP_ENV_NAME="${LVP_ENV_NAME:-ei_world_model}"
LVP_TUNED_CKPT="${LVP_TUNED_CKPT:-${MODEL_BASE}/LVP/data/ckpts/lvp_14B.ckpt}"
LVP_WAN21_DIR="${LVP_WAN21_DIR:-${MODEL_BASE}/Wan2.1-I2V-14B-480P}"
LVP_DATASET="${LVP_DATASET:-ours_test}"
LVP_FPS="${LVP_FPS:-16}"
LVP_N_FRAMES="${LVP_N_FRAMES:-49}"
LVP_HEIGHT="${LVP_HEIGHT:-480}"
LVP_WIDTH="${LVP_WIDTH:-832}"
LVP_LIMIT_BATCH="${LVP_LIMIT_BATCH:-null}"
LVP_LANG_GUIDANCE="${LVP_LANG_GUIDANCE:-2.5}"
LVP_HIST_GUIDANCE_I2V="${LVP_HIST_GUIDANCE_I2V:-1.5}"
LVP_HIST_GUIDANCE_T2V="${LVP_HIST_GUIDANCE_T2V:-0.0}"

CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"
PROGRESS_LOG="${PROGRESS_LOG:-${RUN_ROOT}/progress_${SLURM_JOB_ID:-local}.log}"

log() {
  echo
  echo "==== $* ===="
}

ts() {
  date -Is
}

progress() {
  local msg="$*"
  local line="[$(ts)] ${msg}"
  echo "${line}"
  echo "${line}" >> "${PROGRESS_LOG}"
}

ckpt_dir_is_ready() {
  local ckpt_dir="$1"
  [[ -d "${ckpt_dir}" ]] || return 1
  local first_entry
  first_entry="$(find "${ckpt_dir}" -mindepth 1 -maxdepth 2 -print -quit 2>/dev/null || true)"
  [[ -n "${first_entry}" ]]
}

manifest_expected_count() {
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
        json.loads(line)
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
print(count)
PY
}

count_output_videos() {
  local outputs_dir="$1"
  if [[ ! -d "${outputs_dir}" ]]; then
    echo 0
    return
  fi
  find "${outputs_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

run_cmd_with_heartbeat() {
  local label="$1"
  local log_file="$2"
  shift 2
  local -a cmd=("$@")

  local start_epoch
  start_epoch="$(date +%s)"
  "${cmd[@]}" >>"${log_file}" 2>&1 &
  local cmd_pid="$!"

  while kill -0 "${cmd_pid}" >/dev/null 2>&1; do
    sleep "${HEARTBEAT_SECS}"
    if kill -0 "${cmd_pid}" >/dev/null 2>&1; then
      local now_epoch elapsed
      now_epoch="$(date +%s)"
      elapsed=$((now_epoch - start_epoch))
      progress "HEARTBEAT ${label}: still running elapsed=${elapsed}s log=${log_file}"
    fi
  done

  set +e
  wait "${cmd_pid}"
  local status="$?"
  set -e
  return "${status}"
}

if [[ -z "${DATASET_NAMES}" ]]; then
  echo "DATASET_NAMES is required (comma-separated). Example: DATASET_NAMES=physics_iq,wisa80k" >&2
  exit 1
fi

if [[ "${MISSING_CKPT_ACTION}" != "skip" && "${MISSING_CKPT_ACTION}" != "fail" ]]; then
  echo "Invalid MISSING_CKPT_ACTION=${MISSING_CKPT_ACTION}; expected skip|fail." >&2
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

mkdir -p "$(dirname "${PROGRESS_LOG}")"
{
  echo "[$(ts)] START run_id=${RUN_ID} run_root=${RUN_ROOT}"
  echo "[$(ts)] CONFIG datasets=${DATASET_NAMES} targets=${TARGETS} tasks=${TASKS} max_samples=${MAX_SAMPLES} skip_existing=${SKIP_EXISTING} continue_on_error=${CONTINUE_ON_ERROR} heartbeat_secs=${HEARTBEAT_SECS} missing_ckpt_action=${MISSING_CKPT_ACTION}"
} >> "${PROGRESS_LOG}"

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

activate_env_name() {
  local env_name="$1"
  conda activate "${env_name}"
}

prepare_lvp_ckpts() {
  local lvp_root="${PROJECT_DIR}/third_party/large-video-planner"
  local ckpt_root="${lvp_root}/data/ckpts"
  mkdir -p "${ckpt_root}"

  if [[ ! -e "${ckpt_root}/lvp_14B.ckpt" ]]; then
    if [[ -f "${LVP_TUNED_CKPT}" ]]; then
      ln -s "${LVP_TUNED_CKPT}" "${ckpt_root}/lvp_14B.ckpt"
    else
      echo "LVP tuned checkpoint missing: ${LVP_TUNED_CKPT}" >&2
      return 1
    fi
  fi

  if [[ ! -e "${ckpt_root}/Wan2.1-I2V-14B-480P" ]]; then
    if [[ -d "${LVP_WAN21_DIR}" ]]; then
      ln -s "${LVP_WAN21_DIR}" "${ckpt_root}/Wan2.1-I2V-14B-480P"
    else
      echo "Wan2.1 base checkpoint dir missing for LVP: ${LVP_WAN21_DIR}" >&2
      return 1
    fi
  fi

  return 0
}

manifest_rows() {
  local manifest_path="$1"
  local max_samples="$2"
  python - "$manifest_path" "$max_samples" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
max_samples = int(sys.argv[2])

# Use a non-whitespace delimiter so empty fields (e.g., t2v image_path) are
# preserved when parsed by bash `read`.
DELIM = "\x1f"

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
        print(DELIM.join([sample_id, prompt, image_path, output_video]))
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
PY
}

run_wan22_manifest() {
  local dataset_name="$1"
  local task="$2"
  local manifest_path="${RUN_ROOT}/wan22/${dataset_name}/${task}/inputs/manifest.jsonl"
  local outputs_dir="${RUN_ROOT}/wan22/${dataset_name}/${task}/outputs"

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

  if ! ckpt_dir_is_ready "${ckpt_dir}"; then
    local msg="wan22/${dataset_name}/${task}: checkpoint dir missing/empty (${ckpt_dir})"
    echo "${msg}" >&2
    progress "SLICE SKIP ${msg}"
    if [[ "${MISSING_CKPT_ACTION}" == "fail" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
    return 0
  fi

  local expected_total outputs_before
  expected_total="$(manifest_expected_count "${manifest_path}" "${MAX_SAMPLES}")"
  outputs_before="$(count_output_videos "${outputs_dir}")"
  progress "SLICE START wan22/${dataset_name}/${task}: expected=${expected_total} outputs_before=${outputs_before}"

  local processed_count=0
  local sample_count=0
  local success_count=0
  local failed_count=0
  local skipped_existing=0
  local skipped_missing_input=0
  while IFS=$'\x1f' read -r sample_id prompt image_path output_video; do
    [[ -z "${sample_id}" ]] && continue
    processed_count=$((processed_count + 1))
    if [[ "${SKIP_EXISTING}" == "1" && -s "${output_video}" ]]; then
      skipped_existing=$((skipped_existing + 1))
      progress "wan22/${dataset_name}/${task}: skip_existing sample=${processed_count}/${expected_total} sample_id=${sample_id}"
      continue
    fi
    sample_count=$((sample_count + 1))

    local log_file="${RUN_ROOT}/wan22/${dataset_name}/${task}/logs/${sample_id}.log"
    mkdir -p "$(dirname "${output_video}")" "$(dirname "${log_file}")"
    local sample_start_epoch
    sample_start_epoch="$(date +%s)"

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
        skipped_missing_input=$((skipped_missing_input + 1))
        progress "wan22/${dataset_name}/${task}: skip_missing_input sample=${processed_count}/${expected_total} sample_id=${sample_id} image_path=${image_path}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi
      cmd+=(--image "${image_path}")
    fi

    progress "wan22/${dataset_name}/${task}: start sample=${processed_count}/${expected_total} sample_id=${sample_id} output=${output_video}"
    {
      echo "[$(date -Is)] sample_id=${sample_id}"
      echo "[$(date -Is)] output=${output_video}"
      echo "[$(date -Is)] cmd=${cmd[*]}"
    } >"${log_file}"

    if ! run_cmd_with_heartbeat "wan22/${dataset_name}/${task}/${sample_id}" "${log_file}" "${cmd[@]}"; then
      echo "wan22/${dataset_name}/${task}/${sample_id} failed. See ${log_file}" >&2
      failed_count=$((failed_count + 1))
      local sample_end_epoch elapsed
      sample_end_epoch="$(date +%s)"
      elapsed=$((sample_end_epoch - sample_start_epoch))
      progress "wan22/${dataset_name}/${task}: FAILED sample=${processed_count}/${expected_total} sample_id=${sample_id} elapsed=${elapsed}s log=${log_file}"
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    else
      success_count=$((success_count + 1))
      local sample_end_epoch elapsed
      sample_end_epoch="$(date +%s)"
      elapsed=$((sample_end_epoch - sample_start_epoch))
      progress "wan22/${dataset_name}/${task}: done sample=${processed_count}/${expected_total} sample_id=${sample_id} elapsed=${elapsed}s output_exists=$([[ -s "${output_video}" ]] && echo 1 || echo 0)"
    fi
  done < <(manifest_rows "${manifest_path}" "${MAX_SAMPLES}")

  local outputs_after
  outputs_after="$(count_output_videos "${outputs_dir}")"
  progress "SLICE DONE wan22/${dataset_name}/${task}: expected=${expected_total} processed=${processed_count} attempted=${sample_count} success=${success_count} failed=${failed_count} skipped_existing=${skipped_existing} skipped_missing_input=${skipped_missing_input} outputs_after=${outputs_after}"
  log "Finished wan22/${dataset_name}/${task} (${sample_count} samples, skipped_existing=${skipped_existing})"
}

run_wan21_manifest() {
  local dataset_name="$1"
  local task="$2"
  local manifest_path="${RUN_ROOT}/wan21/${dataset_name}/${task}/inputs/manifest.jsonl"
  local outputs_dir="${RUN_ROOT}/wan21/${dataset_name}/${task}/outputs"

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

  if ! ckpt_dir_is_ready "${ckpt_dir}"; then
    local msg="wan21/${dataset_name}/${task}: checkpoint dir missing/empty (${ckpt_dir})"
    echo "${msg}" >&2
    progress "SLICE SKIP ${msg}"
    if [[ "${MISSING_CKPT_ACTION}" == "fail" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
    return 0
  fi

  local expected_total outputs_before
  expected_total="$(manifest_expected_count "${manifest_path}" "${MAX_SAMPLES}")"
  outputs_before="$(count_output_videos "${outputs_dir}")"
  progress "SLICE START wan21/${dataset_name}/${task}: expected=${expected_total} outputs_before=${outputs_before}"

  local processed_count=0
  local sample_count=0
  local success_count=0
  local failed_count=0
  local skipped_existing=0
  local skipped_missing_input=0
  while IFS=$'\x1f' read -r sample_id prompt image_path output_video; do
    [[ -z "${sample_id}" ]] && continue
    processed_count=$((processed_count + 1))
    if [[ "${SKIP_EXISTING}" == "1" && -s "${output_video}" ]]; then
      skipped_existing=$((skipped_existing + 1))
      progress "wan21/${dataset_name}/${task}: skip_existing sample=${processed_count}/${expected_total} sample_id=${sample_id}"
      continue
    fi
    sample_count=$((sample_count + 1))

    local log_file="${RUN_ROOT}/wan21/${dataset_name}/${task}/logs/${sample_id}.log"
    mkdir -p "$(dirname "${output_video}")" "$(dirname "${log_file}")"
    local sample_start_epoch
    sample_start_epoch="$(date +%s)"

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
        skipped_missing_input=$((skipped_missing_input + 1))
        progress "wan21/${dataset_name}/${task}: skip_missing_input sample=${processed_count}/${expected_total} sample_id=${sample_id} image_path=${image_path}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi
      cmd+=(--image "${image_path}")
    fi

    progress "wan21/${dataset_name}/${task}: start sample=${processed_count}/${expected_total} sample_id=${sample_id} output=${output_video}"
    {
      echo "[$(date -Is)] sample_id=${sample_id}"
      echo "[$(date -Is)] output=${output_video}"
      echo "[$(date -Is)] cmd=${cmd[*]}"
    } >"${log_file}"

    if ! run_cmd_with_heartbeat "wan21/${dataset_name}/${task}/${sample_id}" "${log_file}" "${cmd[@]}"; then
      echo "wan21/${dataset_name}/${task}/${sample_id} failed. See ${log_file}" >&2
      failed_count=$((failed_count + 1))
      local sample_end_epoch elapsed
      sample_end_epoch="$(date +%s)"
      elapsed=$((sample_end_epoch - sample_start_epoch))
      progress "wan21/${dataset_name}/${task}: FAILED sample=${processed_count}/${expected_total} sample_id=${sample_id} elapsed=${elapsed}s log=${log_file}"
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    else
      success_count=$((success_count + 1))
      local sample_end_epoch elapsed
      sample_end_epoch="$(date +%s)"
      elapsed=$((sample_end_epoch - sample_start_epoch))
      progress "wan21/${dataset_name}/${task}: done sample=${processed_count}/${expected_total} sample_id=${sample_id} elapsed=${elapsed}s output_exists=$([[ -s "${output_video}" ]] && echo 1 || echo 0)"
    fi
  done < <(manifest_rows "${manifest_path}" "${MAX_SAMPLES}")

  local outputs_after
  outputs_after="$(count_output_videos "${outputs_dir}")"
  progress "SLICE DONE wan21/${dataset_name}/${task}: expected=${expected_total} processed=${processed_count} attempted=${sample_count} success=${success_count} failed=${failed_count} skipped_existing=${skipped_existing} skipped_missing_input=${skipped_missing_input} outputs_after=${outputs_after}"
  log "Finished wan21/${dataset_name}/${task} (${sample_count} samples, skipped_existing=${skipped_existing})"
}

run_lvp_manifest() {
  local dataset_name="$1"
  local task="$2"
  local manifest_path="${RUN_ROOT}/lvp/${dataset_name}/${task}/inputs/manifest.jsonl"
  local outputs_dir="${RUN_ROOT}/lvp/${dataset_name}/${task}/outputs"

  if [[ ! -f "${manifest_path}" ]]; then
    log "Skipping lvp/${dataset_name}/${task}: manifest missing (${manifest_path})"
    return 0
  fi

  local expected_total outputs_before
  expected_total="$(manifest_expected_count "${manifest_path}" "${MAX_SAMPLES}")"
  outputs_before="$(count_output_videos "${outputs_dir}")"
  progress "SLICE START lvp/${dataset_name}/${task}: expected=${expected_total} outputs_before=${outputs_before}"

  local runtime_dir="${RUN_ROOT}/lvp/${dataset_name}/${task}/runtime"
  local metadata_csv="${runtime_dir}/metadata.csv"
  local lvp_log="${RUN_ROOT}/lvp/${dataset_name}/${task}/logs/lvp_run.log"
  local hydra_dir="${RUN_ROOT}/lvp/${dataset_name}/${task}/logs/hydra_${RUN_ID}"
  mkdir -p "${runtime_dir}" "$(dirname "${lvp_log}")" "${hydra_dir}"

  if ! python - "$manifest_path" "$metadata_csv" "$task" "$MAX_SAMPLES" "$LVP_HEIGHT" "$LVP_WIDTH" "$LVP_N_FRAMES" "$LVP_FPS" "$SKIP_EXISTING" <<'PY'
import csv
import json
import os
import sys

manifest_path, output_csv, task, max_samples, height, width, n_frames, fps, skip_existing = sys.argv[1:]
max_samples = int(max_samples)
skip_existing = (skip_existing == "1")

rows = []
skipped = 0
skipped_existing = 0
with open(manifest_path, "r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)

        prompt = str(row.get("prompt", "")).strip()
        output_video = str(row.get("output_video", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()

        # Prefer explicit image_path; fallback to shared sample metadata input_image.
        image_path = str(row.get("image_path", "")).strip()
        if not image_path:
            metadata_path = str(row.get("metadata_path", "")).strip()
            if metadata_path and os.path.isfile(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as m:
                    metadata = json.load(m)
                image_path = str(metadata.get("input_image", "")).strip()

        if not prompt or not output_video or not image_path or not os.path.isfile(image_path):
            skipped += 1
            continue

        if skip_existing and os.path.isfile(output_video) and os.path.getsize(output_video) > 0:
            skipped_existing += 1
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "video_path": image_path,
                "caption": prompt,
                "height": int(height),
                "width": int(width),
                "n_frames": int(n_frames),
                "fps": int(fps),
                "split": "validation",
                "output_video": output_video,
            }
        )
        if max_samples > 0 and len(rows) >= max_samples:
            break

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sample_id",
            "video_path",
            "caption",
            "height",
            "width",
            "n_frames",
            "fps",
            "split",
            "output_video",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote {len(rows)} LVP metadata rows to {output_csv} (skipped={skipped}, skipped_existing={skipped_existing})")
PY
  then
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "lvp/${dataset_name}/${task}: metadata generation failed; continuing due to CONTINUE_ON_ERROR=1" >&2
      return 0
    fi
    exit 1
  fi

  local pending_rows
  pending_rows="$(python - "$metadata_csv" <<'PY'
import csv
import sys

count = 0
with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for _ in reader:
        count += 1
print(count)
PY
)"
  if [[ "${pending_rows}" -eq 0 ]]; then
    progress "SLICE DONE lvp/${dataset_name}/${task}: expected=${expected_total} pending=0 skipped_existing_or_filtered=all outputs_after=$(count_output_videos "${outputs_dir}")"
    log "Skipping lvp/${dataset_name}/${task}: no pending samples (all outputs already exist)"
    return 0
  fi

  local hist_guidance="${LVP_HIST_GUIDANCE_I2V}"
  if [[ "${task}" == "t2v" ]]; then
    # LVP is image-conditioned; for t2v slots we reduce history guidance toward text-dominant generation.
    hist_guidance="${LVP_HIST_GUIDANCE_T2V}"
  elif [[ "${task}" != "i2v" ]]; then
    echo "Unknown task '${task}' for lvp. Use t2v or i2v." >&2
    exit 1
  fi

  {
    echo "[$(date -Is)] dataset=${dataset_name} task=${task}"
    echo "[$(date -Is)] metadata_csv=${metadata_csv}"
    echo "[$(date -Is)] hydra_dir=${hydra_dir}"
  } >"${lvp_log}"

  local -a lvp_cmd=(
    python -m main
    +name="lvp_layout_${dataset_name}_${task}_${RUN_ID}"
    experiment=exp_video
    experiment.strategy=auto
    algorithm=wan_i2v
    "dataset=${LVP_DATASET}"
    "experiment.tasks=[validation]"
    algorithm.logging.video_type=single
    experiment.num_nodes=1
    "experiment.validation.limit_batch=${LVP_LIMIT_BATCH}"
    "algorithm.hist_guidance=${hist_guidance}"
    "algorithm.lang_guidance=${LVP_LANG_GUIDANCE}"
    dataset.data_root=/
    "dataset.metadata_path=${metadata_csv}"
    "dataset.height=${LVP_HEIGHT}"
    "dataset.width=${LVP_WIDTH}"
    "dataset.n_frames=${LVP_N_FRAMES}"
    "dataset.fps=${LVP_FPS}"
    dataset.filtering.disable=true
    dataset.test_percentage=1.0
    dataset.load_prompt_embed=false
    dataset.check_video_path=false
    "hydra.run.dir=${hydra_dir}"
  )
  progress "lvp/${dataset_name}/${task}: start validation pending=${pending_rows} metadata=${metadata_csv}"
  if ! run_cmd_with_heartbeat "lvp/${dataset_name}/${task}" "${lvp_log}" "${lvp_cmd[@]}"; then
    echo "lvp/${dataset_name}/${task} failed. See ${lvp_log}" >&2
    progress "lvp/${dataset_name}/${task}: FAILED run log=${lvp_log}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi

  local expected=0
  local missing=0
  read -r expected missing < <(python - "$metadata_csv" <<'PY'
import csv
import os
import sys

expected = 0
missing = 0
with open(sys.argv[1], "r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        output_video = str(row.get("output_video", "")).strip()
        if not output_video:
            continue
        expected += 1
        if not os.path.isfile(output_video):
            missing += 1

print(expected, missing)
PY
  )

  if [[ "${missing}" -gt 0 ]]; then
    echo "lvp/${dataset_name}/${task}: ${missing}/${expected} expected outputs missing after run" >&2
    progress "lvp/${dataset_name}/${task}: completed with missing_outputs=${missing}/${expected}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi

  local outputs_after
  outputs_after="$(count_output_videos "${outputs_dir}")"
  progress "SLICE DONE lvp/${dataset_name}/${task}: expected=${expected_total} pending=${pending_rows} checked=${expected} missing=${missing} outputs_after=${outputs_after}"
  log "Finished lvp/${dataset_name}/${task} (${expected} samples checked)"
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
    lvp)
      log "Running LVP layout batch"
      activate_env_name "${LVP_ENV_NAME}"
      export WANDB_MODE="${WANDB_MODE:-offline}"
      if ! prepare_lvp_ckpts; then
        progress "TARGET SKIP lvp: required checkpoints unavailable (LVP_TUNED_CKPT=${LVP_TUNED_CKPT}, LVP_WAN21_DIR=${LVP_WAN21_DIR})"
        if [[ "${MISSING_CKPT_ACTION}" == "fail" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
          exit 1
        fi
        continue
      fi
      cd "${PROJECT_DIR}/third_party/large-video-planner"
      for raw_dataset in "${dataset_list[@]}"; do
        dataset_name="${raw_dataset//[[:space:]]/}"
        [[ -z "${dataset_name}" ]] && continue
        for raw_task in "${task_list[@]}"; do
          task="${raw_task//[[:space:]]/}"
          [[ -z "${task}" ]] && continue
          run_lvp_manifest "${dataset_name}" "${task}"
        done
      done
      ;;
    "")
      ;;
    *)
      echo "Unknown target '${target}'. Supported layout targets: wan22,wan21,lvp" >&2
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
echo "progress_log=${PROGRESS_LOG}"
