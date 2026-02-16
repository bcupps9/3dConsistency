#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-bcupps}"
REMOTE_HOST="${REMOTE_HOST:-login.rc.fas.harvard.edu}"
REMOTE_DIR="${REMOTE_DIR:-~/projects/3dConsistency}"

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "runs" \
  --exclude "__pycache__" \
  "${PWD}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
