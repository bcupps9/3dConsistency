#!/usr/bin/env bash
# Refresh access/modification timestamps on all files in scratch so the
# filesystem purge policy doesn't delete them.
# Run via cron: 0 0 1 * * /n/home12/bcupps/projects/3dConsistency/scripts/touch_scratch.sh
set -euo pipefail

SCRATCH_DIR="/n/netscratch/ydu_lab/Lab/bcupps"
LOG_FILE="/n/home12/bcupps/projects/3dConsistency/scripts/touch_scratch.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting touch of ${SCRATCH_DIR}" >> "${LOG_FILE}"

find "${SCRATCH_DIR}" -not -type l -exec touch -c {} + 2>/dev/null

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" >> "${LOG_FILE}"
