#!/usr/bin/env bash
#SBATCH --job-name=hf_dataset_dl
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=hf_download-%j.out

set -euo pipefail

# Usage example:
#   sbatch \
#     -p serial_requeue \
#     --cpus-per-task=8 \
#     --mem=32G \
#     --time=48:00:00 \
#     --output=/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/download-%j.out \
#     --export=ALL,REPO_ID=qihoo360/WISA-80K,REPO_TYPE=dataset,DEST=/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K,ALLOW_PATTERNS='data/**',HF_TOKEN_FILE=/n/home12/bcupps/hf_key.txt,HF_ENV_NAME=hf_dl \
#     scripts/slurm_download_hf_dataset.sh

REPO_ID="${REPO_ID:-qihoo360/WISA-80K}"
REPO_TYPE="${REPO_TYPE:-dataset}"
DEST="${DEST:-/n/netscratch/ydu_lab/Lab/bcupps/datasets/raw/WISA-80K}"
ALLOW_PATTERNS="${ALLOW_PATTERNS:-data/**}" # comma-separated patterns

# Auth: prefer HF_TOKEN env, otherwise read from HF_TOKEN_FILE.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/hf_key.txt}"

# Env/bootstrap
HF_ENV_NAME="${HF_ENV_NAME:-hf_dl}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-Miniforge3/24.11.3-fasrc02}"
CONDA_HOOK_BIN="${CONDA_HOOK_BIN:-/n/sw/Miniforge3-24.11.3-0-fasrc02/bin/conda}"
INSTALL_DL_DEPS="${INSTALL_DL_DEPS:-1}"

log() {
  echo
  echo "==== $* ===="
}

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load "${CUDA_MODULE}" || true
  module load "${MINIFORGE_MODULE}" || true
fi

if [[ -x "${CONDA_HOOK_BIN}" ]]; then
  eval "$("${CONDA_HOOK_BIN}" shell.bash hook)"
  conda activate "${HF_ENV_NAME}"
fi

if [[ "${INSTALL_DL_DEPS}" == "1" ]]; then
  log "Installing downloader dependencies"
  python -m pip install -U huggingface_hub hf_xet
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HF_TOKEN
    HF_TOKEN="$(<"${HF_TOKEN_FILE}")"
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF token not set. Export HF_TOKEN or provide HF_TOKEN_FILE." >&2
  exit 1
fi

mkdir -p "${DEST}"

log "Download request"
echo "repo_id=${REPO_ID}"
echo "repo_type=${REPO_TYPE}"
echo "dest=${DEST}"
echo "allow_patterns=${ALLOW_PATTERNS}"
echo "host=$(hostname -f || hostname)"
echo "started=$(date -Is)"

python - <<'PY'
import os
from huggingface_hub import HfApi, snapshot_download

repo_id = os.environ["REPO_ID"]
repo_type = os.environ["REPO_TYPE"]
dest = os.environ["DEST"]
token = os.environ.get("HF_TOKEN")
allow_patterns_raw = os.environ.get("ALLOW_PATTERNS", "").strip()
allow_patterns = [p.strip() for p in allow_patterns_raw.split(",") if p.strip()]

kwargs = dict(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=dest,
    token=token,
)
if allow_patterns:
    kwargs["allow_patterns"] = allow_patterns

print("snapshot_download kwargs:", {k: v for k, v in kwargs.items() if k != "token"})
snapshot_path = snapshot_download(**kwargs)
print("snapshot_path:", snapshot_path)

api = HfApi(token=token)
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
print("remote_file_count:", len(files))
print("remote_file_sample:", files[:25])
PY

log "Local download summary"
du -sh "${DEST}" || true
echo "local_file_count=$(find "${DEST}" -type f | wc -l | tr -d ' ')"
echo "local_mp4_count=$(find "${DEST}" -type f -name '*.mp4' | wc -l | tr -d ' ')"
echo "local_zip_count=$(find "${DEST}" -type f -name '*.zip' | wc -l | tr -d ' ')"

echo "first_mp4_files:"
find "${DEST}" -type f -name '*.mp4' | head -n 20 || true
echo "first_zip_files:"
find "${DEST}" -type f -name '*.zip' | head -n 20 || true

echo "finished=$(date -Is)"
