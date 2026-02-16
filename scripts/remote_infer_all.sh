#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/3dConsistency}"

# Comma-separated list of targets to run in sequence.
# Supported values: wan22,wan21,lvp
TARGETS="${TARGETS:-wan22,wan21,lvp}"

# lvp mode:
# - smoke: fast installation check using wan_toy + dummy dataset
# - full:  wan_i2v inference path (expects dataset/checkpoints configured)
RUN_MODE="${RUN_MODE:-smoke}"

SCRATCH_BASE_DEFAULT="/n/netscratch/ydu_lab/Lab/bcupps"
SCRATCH_BASE="${SCRATCH_BASE:-$SCRATCH_BASE_DEFAULT}"
PROJECT_RUN_DIR="${PROJECT_DIR}/runs/${RUN_ID}"
SCRATCH_RUN_DIR="${SCRATCH_BASE}/results/${RUN_ID}"

MODEL_BASE="${MODEL_BASE:-/n/netscratch/ydu_lab/Lab/bcupps/models}"

WAN22_ENV="${WAN22_ENV:-${PROJECT_DIR}/.mamba/wan2}"
WAN22_TASK="${WAN22_TASK:-t2v-A14B}"
WAN22_SIZE="${WAN22_SIZE:-1280*720}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${MODEL_BASE}/Wan2.2-T2V-A14B}"
WAN22_IMAGE="${WAN22_IMAGE:-${PROJECT_DIR}/third_party/Wan2.2/examples/i2v_input.JPG}"
WAN22_PROMPT="${WAN22_PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"

WAN21_ENV="${WAN21_ENV:-${PROJECT_DIR}/.mamba/wan21}"
WAN21_TASK="${WAN21_TASK:-i2v-14B}"
WAN21_SIZE="${WAN21_SIZE:-832*480}"
WAN21_CKPT_DIR="${WAN21_CKPT_DIR:-${MODEL_BASE}/Wan2.1-I2V-14B-480P}"
WAN21_IMAGE="${WAN21_IMAGE:-${PROJECT_DIR}/third_party/Wan2.1/examples/i2v_input.JPG}"
WAN21_PROMPT="${WAN21_PROMPT:-Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.}"

LVP_ENV_NAME="${LVP_ENV_NAME:-ei_world_model}"
LVP_DATASET="${LVP_DATASET:-ours_test}"
LVP_LIMIT_BATCH="${LVP_LIMIT_BATCH:-null}"
LVP_HIST_GUIDANCE="${LVP_HIST_GUIDANCE:-1.5}"
LVP_LANG_GUIDANCE="${LVP_LANG_GUIDANCE:-2.5}"
LVP_WAN21_DIR="${LVP_WAN21_DIR:-${MODEL_BASE}/Wan2.1-I2V-14B-480P}"
LVP_TUNED_CKPT="${LVP_TUNED_CKPT:-${MODEL_BASE}/LVP/data/ckpts/lvp_14B.ckpt}"

CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"

log() {
  echo
  echo "==== $* ===="
}

model_output_dir() {
  local model_key="$1"
  echo "${RUN_DIR}/${model_key}"
}

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
  echo "Invalid RUN_MODE=${RUN_MODE}; expected 'smoke' or 'full'." >&2
  exit 1
fi

RUN_DIR="${PROJECT_RUN_DIR}"
mkdir -p "${PROJECT_RUN_DIR}"
if [[ -d "${SCRATCH_BASE}" ]]; then
  mkdir -p "${SCRATCH_RUN_DIR}"
  RUN_DIR="${SCRATCH_RUN_DIR}"
fi

cd "${PROJECT_DIR}"

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

run_wan22() {
  log "Running Wan2.2 inference"
  activate_env_path "${WAN22_ENV}"

  if [[ "${WAN22_TASK}" != "t2v-A14B" && "${WAN22_TASK}" != "i2v-A14B" ]]; then
    echo "Wan2.2 runner only supports t2v-A14B or i2v-A14B. Got: ${WAN22_TASK}" >&2
    exit 1
  fi

  if [[ ! -d "${WAN22_CKPT_DIR}" ]]; then
    echo "Wan2.2 checkpoint dir not found: ${WAN22_CKPT_DIR}" >&2
    exit 1
  fi

  local model_dir
  model_dir="$(model_output_dir wan22)"
  mkdir -p "${model_dir}"
  local save_file="${model_dir}/${WAN22_TASK}_${RUN_ID}.mp4"
  cd "${PROJECT_DIR}/third_party/Wan2.2"
  if [[ "${WAN22_TASK}" == "i2v-A14B" ]]; then
    if [[ ! -f "${WAN22_IMAGE}" ]]; then
      echo "Wan2.2 image not found: ${WAN22_IMAGE}" >&2
      exit 1
    fi
    python generate.py \
      --task "${WAN22_TASK}" \
      --size "${WAN22_SIZE}" \
      --ckpt_dir "${WAN22_CKPT_DIR}" \
      --image "${WAN22_IMAGE}" \
      --offload_model True \
      --convert_model_dtype \
      --prompt "${WAN22_PROMPT}" \
      --save_file "${save_file}"
  else
    python generate.py \
      --task "${WAN22_TASK}" \
      --size "${WAN22_SIZE}" \
      --ckpt_dir "${WAN22_CKPT_DIR}" \
      --offload_model True \
      --convert_model_dtype \
      --prompt "${WAN22_PROMPT}" \
      --save_file "${save_file}"
  fi
}

run_wan21() {
  log "Running Wan2.1 inference"
  activate_env_path "${WAN21_ENV}"

  if [[ "${WAN21_TASK}" != "t2v-14B" && "${WAN21_TASK}" != "i2v-14B" ]]; then
    echo "Wan2.1 runner only supports t2v-14B or i2v-14B. Got: ${WAN21_TASK}" >&2
    exit 1
  fi

  if [[ ! -d "${WAN21_CKPT_DIR}" ]]; then
    echo "Wan2.1 checkpoint dir not found: ${WAN21_CKPT_DIR}" >&2
    exit 1
  fi
  if [[ "${WAN21_TASK}" == "i2v-14B" && ! -f "${WAN21_IMAGE}" ]]; then
    echo "Wan2.1 image not found: ${WAN21_IMAGE}" >&2
    exit 1
  fi

  local model_dir
  model_dir="$(model_output_dir wan21)"
  mkdir -p "${model_dir}"
  local save_file="${model_dir}/${WAN21_TASK}_${RUN_ID}.mp4"
  cd "${PROJECT_DIR}/third_party/Wan2.1"
  if [[ "${WAN21_TASK}" == "i2v-14B" ]]; then
    python generate.py \
      --task "${WAN21_TASK}" \
      --size "${WAN21_SIZE}" \
      --ckpt_dir "${WAN21_CKPT_DIR}" \
      --image "${WAN21_IMAGE}" \
      --offload_model True \
      --t5_cpu \
      --prompt "${WAN21_PROMPT}" \
      --save_file "${save_file}"
  else
    python generate.py \
      --task "${WAN21_TASK}" \
      --size "${WAN21_SIZE}" \
      --ckpt_dir "${WAN21_CKPT_DIR}" \
      --offload_model True \
      --t5_cpu \
      --prompt "${WAN21_PROMPT}" \
      --save_file "${save_file}"
  fi
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
      exit 1
    fi
  fi

  if [[ ! -e "${ckpt_root}/Wan2.1-I2V-14B-480P" ]]; then
    if [[ -d "${LVP_WAN21_DIR}" ]]; then
      ln -s "${LVP_WAN21_DIR}" "${ckpt_root}/Wan2.1-I2V-14B-480P"
    else
      echo "Wan2.1 base checkpoint dir missing for LVP: ${LVP_WAN21_DIR}" >&2
      exit 1
    fi
  fi
}

run_lvp() {
  log "Running LVP (${RUN_MODE})"
  activate_env_name "${LVP_ENV_NAME}"

  export WANDB_MODE="${WANDB_MODE:-offline}"
  cd "${PROJECT_DIR}/third_party/large-video-planner"

  local model_dir
  model_dir="$(model_output_dir lvp)"
  mkdir -p "${model_dir}"

  if [[ "${RUN_MODE}" == "smoke" ]]; then
    python -m main \
      +name="lvp_smoke_${RUN_ID}" \
      experiment=exp_video \
      algorithm=wan_toy \
      dataset=dummy \
      'experiment.tasks=[validation]' \
      experiment.validation.limit_batch=1 \
      "hydra.run.dir=${model_dir}/smoke"
  else
    prepare_lvp_ckpts
    python -m main \
      +name="lvp_i2v_${RUN_ID}" \
      experiment=exp_video \
      algorithm=wan_i2v \
      "dataset=${LVP_DATASET}" \
      'experiment.tasks=[validation]' \
      algorithm.logging.video_type=single \
      experiment.num_nodes=1 \
      "experiment.validation.limit_batch=${LVP_LIMIT_BATCH}" \
      "algorithm.hist_guidance=${LVP_HIST_GUIDANCE}" \
      "algorithm.lang_guidance=${LVP_LANG_GUIDANCE}" \
      "hydra.run.dir=${model_dir}/full"
  fi
}

IFS=',' read -r -a target_list <<< "${TARGETS}"

# Create model-level directories up front so a run always has predictable structure.
for raw_target in "${target_list[@]}"; do
  target="${raw_target//[[:space:]]/}"
  case "${target}" in
    wan22|wan21|lvp)
      mkdir -p "$(model_output_dir "${target}")"
      ;;
    "")
      ;;
    *)
      # Invalid targets are handled in the execution loop below.
      ;;
  esac
done

for raw_target in "${target_list[@]}"; do
  target="${raw_target//[[:space:]]/}"
  case "${target}" in
    wan22)
      run_wan22
      ;;
    wan21)
      run_wan21
      ;;
    lvp)
      run_lvp
      ;;
    "")
      ;;
    *)
      echo "Unknown target '${target}'. Use wan22, wan21, lvp." >&2
      exit 1
      ;;
  esac
done

if [[ "${RUN_DIR}" != "${PROJECT_RUN_DIR}" ]]; then
  log "Syncing outputs back to project runs dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -az "${RUN_DIR}/" "${PROJECT_RUN_DIR}/"
  else
    cp -R "${RUN_DIR}/" "${PROJECT_RUN_DIR}/"
  fi
fi

log "All requested targets finished"
echo "run_id=${RUN_ID}"
echo "outputs=${PROJECT_RUN_DIR}"
