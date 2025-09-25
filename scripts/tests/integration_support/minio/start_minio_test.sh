#!/usr/bin/env bash
set -euo pipefail

# Start a local MinIO server suitable for S3-compatible integration tests.
# Requires: Docker with compose plugin (docker compose ...)
# Usage: ./start_minio_test.sh [bucket-name]
# Defaults: bucket name = ingest-test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.minio-test.yaml"
CONTAINER_NAME="minio-test"
BUCKET_NAME=${1:-ingest-test}
TIMEOUT_SECS=${TIMEOUT_SECS:-120}

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi

echo "[minio-test] Starting docker compose stack..."
docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans

# Wait for health status
start_ts=$(date +%s)
while true; do
  status=$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
  if [[ "${status}" == "healthy" ]]; then
    echo "[minio-test] MinIO is healthy at http://localhost:9000 (console http://localhost:9001)"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT_SECS )); then
    echo "ERROR: Timed out waiting for MinIO to become healthy (>${TIMEOUT_SECS}s)" >&2
    docker compose -f "${COMPOSE_FILE}" logs --no-color || true
    exit 2
  fi
  echo "[minio-test] Waiting for health... (current: ${status})"
  sleep 3
done

# Create bucket using MinIO client (mc)
echo "[minio-test] Ensuring bucket '${BUCKET_NAME}' exists..."
docker pull minio/mc:latest >/dev/null || true
docker run --rm --network host \
  -e MC_HOST_local="http://minioadmin:minioadmin@localhost:9000" \
  minio/mc:latest \
  mb --ignore-existing local/${BUCKET_NAME}

echo "[minio-test] Export these env vars to use in tests:"
echo "  export AWS_ACCESS_KEY_ID=minioadmin"
echo "  export AWS_SECRET_ACCESS_KEY=minioadmin"
echo "  export AWS_ENDPOINT_URL=http://localhost:9000"
echo "  export AWS_DEFAULT_REGION=us-east-1"
echo "  export INGEST_INTEGRATION_TEST_MINIO=http://localhost:9000/${BUCKET_NAME}"
