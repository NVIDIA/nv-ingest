#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the redis-test container and volumes created by docker compose.
# Usage: ./stop_redis_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.redis-test.yaml"

echo "[redis-test] Stopping and removing containers..."
docker compose -f "${COMPOSE_FILE}" down -v

echo "[redis-test] Done."
