#!/usr/bin/env bash
set -euo pipefail

# Start a local HTTP upload test service suitable for integration tests.
# Requires: Docker with compose plugin (docker compose ...)
# Usage: ./start_http_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.http-test.yaml"
CONTAINER_NAME="http-upload-test"
TIMEOUT_SECS=${TIMEOUT_SECS:-180}

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi

echo "[http-test] Building and starting docker compose stack..."
docker compose -f "${COMPOSE_FILE}" up -d --build --remove-orphans

# Wait for health status
start_ts=$(date +%s)
while true; do
  status=$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
  if [[ "${status}" == "healthy" ]]; then
    echo "[http-test] Service is healthy at http://localhost:18080"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT_SECS )); then
    echo "ERROR: Timed out waiting for HTTP service to become healthy (>${TIMEOUT_SECS}s)" >&2
    docker compose -f "${COMPOSE_FILE}" logs --no-color || true
    exit 2
  fi
  echo "[http-test] Waiting for health... (current: ${status})"
  sleep 3
done

cat <<EOF
[http-test] Export this for tests:
  export INGEST_INTEGRATION_TEST_HTTP=http://localhost:18080
EOF
