#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_nightly.sh"
LOG_DIR="${NEMO_RETRIEVER_NIGHTLY_LOG_DIR:-${REPO_ROOT}/nemo_retriever/artifacts}"
TARGET_UTC_HOUR="${NEMO_RETRIEVER_NIGHTLY_UTC_HOUR:-1}"
TARGET_UTC_MINUTE="${NEMO_RETRIEVER_NIGHTLY_UTC_MINUTE:-0}"
LOOP_LOG="${LOG_DIR}/nightly_loop.log"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${RUNNER_SCRIPT}" ]]; then
  echo "Runner script is not executable: ${RUNNER_SCRIPT}" >&2
  exit 1
fi

sleep_until_next_window() {
  TARGET_UTC_HOUR="${TARGET_UTC_HOUR}" TARGET_UTC_MINUTE="${TARGET_UTC_MINUTE}" \
  python3 - <<'PY'
from datetime import datetime, timedelta, timezone
import os

hour = int(os.environ["TARGET_UTC_HOUR"])
minute = int(os.environ["TARGET_UTC_MINUTE"])
now = datetime.now(timezone.utc)
target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int((target - now).total_seconds()))
PY
}

while true; do
  wait_seconds="$(sleep_until_next_window)"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] sleeping ${wait_seconds}s until next nightly window" | tee -a "${LOOP_LOG}"
  sleep "${wait_seconds}"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] starting nightly run" | tee -a "${LOOP_LOG}"
  if ! "${RUNNER_SCRIPT}" >> "${LOOP_LOG}" 2>&1; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] nightly run failed" | tee -a "${LOOP_LOG}"
  else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] nightly run completed" | tee -a "${LOOP_LOG}"
  fi
done
