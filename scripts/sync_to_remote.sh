#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="bcupps"
REMOTE_HOST="holylogin05"
REMOTE_DIR="~/projects/3dConsistency"

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "runs" \
  --exclude "__pycache__" \
  "${PWD}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
