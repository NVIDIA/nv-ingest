#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the minio-test container and volumes created by docker compose.
# Usage: ./stop_minio_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.minio-test.yaml"

echo "[minio-test] Stopping and removing containers..."
docker compose -f "${COMPOSE_FILE}" down -v

echo "[minio-test] Done."
