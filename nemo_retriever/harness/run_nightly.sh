#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${NEMO_RETRIEVER_PYTHON:-${REPO_ROOT}/.retriever/bin/python}"
RUNS_CONFIG="${NEMO_RETRIEVER_NIGHTLY_CONFIG:-${REPO_ROOT}/nemo_retriever/harness/nightly_config.yaml}"
PYTHONPATH_ENTRY="${REPO_ROOT}/nemo_retriever/src"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${RUNS_CONFIG}" ]]; then
  echo "Nightly config not found: ${RUNS_CONFIG}" >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH_ENTRY}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m nemo_retriever.harness.__main__ nightly --runs-config "${RUNS_CONFIG}" "$@"
