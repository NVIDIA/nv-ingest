#!/usr/bin/env bash
set -euo pipefail

# Stop and remove the kafka-test container and network created by docker compose.
# Usage: ./stop_kafka_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.kafka-test.yaml"

echo "[kafka-test] Stopping and removing containers..."
docker compose -f "${COMPOSE_FILE}" down -v

echo "[kafka-test] Done."
