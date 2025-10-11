#!/usr/bin/env bash
set -euo pipefail

# Start a local Redis suitable for integration tests.
# Requires: Docker with compose plugin (docker compose ...)
# Usage: ./start_redis_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.redis-test.yaml"
CONTAINER_NAME="redis-test"
TIMEOUT_SECS=${TIMEOUT_SECS:-120}

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi

echo "[redis-test] Starting docker compose stack..."
docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans

# Wait for health status
start_ts=$(date +%s)
while true; do
  status=$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
  if [[ "${status}" == "healthy" ]]; then
    echo "[redis-test] Redis is healthy at localhost:6379 (db 0)"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT_SECS )); then
    echo "ERROR: Timed out waiting for Redis to become healthy (>${TIMEOUT_SECS}s)" >&2
    docker compose -f "${COMPOSE_FILE}" logs --no-color || true
    exit 2
  fi
  echo "[redis-test] Waiting for health... (current: ${status})"
  sleep 3
done

cat <<EOF
[redis-test] Export this for tests:
  export INGEST_INTEGRATION_TEST_REDIS=localhost:6379/0
EOF
