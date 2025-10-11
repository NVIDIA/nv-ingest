#!/usr/bin/env bash
set -euo pipefail

# Start a single-node Kafka (KRaft) suitable for local integration tests.
# Requires: Docker with compose plugin (docker compose ...)
# Usage: ./start_kafka_test.sh
# Will:
#  - Pull the bitnami/kafka image if not present
#  - Start service using docker compose file in this directory
#  - Wait until the broker is healthy or timeout

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.kafka-test.yaml"
SERVICE_NAME="kafka"
CONTAINER_NAME="kafka-test"
TIMEOUT_SECS=${TIMEOUT_SECS:-120}

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi

# Pull image explicitly (optional, compose will pull as needed)
DOCKER_IMAGE="bitnami/kafka:3.6"
echo "[kafka-test] Pulling image ${DOCKER_IMAGE} (if needed)..."
docker pull "${DOCKER_IMAGE}" >/dev/null || true

# Start service
echo "[kafka-test] Starting docker compose stack..."
docker compose -f "${COMPOSE_FILE}" up -d --remove-orphans

# Wait for health status
start_ts=$(date +%s)
while true; do
  status=$(docker inspect -f '{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
  if [[ "${status}" == "healthy" ]]; then
    echo "[kafka-test] Kafka is healthy at PLAINTEXT://localhost:9092"
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > TIMEOUT_SECS )); then
    echo "ERROR: Timed out waiting for Kafka to become healthy (>${TIMEOUT_SECS}s)" >&2
    docker compose -f "${COMPOSE_FILE}" logs --no-color || true
    exit 2
  fi
  echo "[kafka-test] Waiting for health... (current: ${status})"
  sleep 3
done
