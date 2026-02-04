#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the http-upload-test container and volumes created by docker compose.
# Usage: ./stop_http_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.http-test.yaml"

echo "[http-test] Stopping and removing containers..."
docker compose -f "${COMPOSE_FILE}" down -v

echo "[http-test] Done."
