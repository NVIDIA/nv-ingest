#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fail on errors (-e) and undefined variables (-u)
set -eux

##############################
# Source Validation Script
##############################
BUILD_SCRIPT_BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "${BUILD_SCRIPT_BASE}/scripts/helper_functions.sh"

# Validate environment
validate_conda_build_environment

##############################
# Determine Git Root
##############################
GIT_ROOT=$(determine_git_root)

##############################
# Input Arguments
##############################
OUTPUT_DIR=${1:-"${BUILD_SCRIPT_BASE}/output_conda_channel"}
CONDA_CHANNEL=${2:-""}
BUILD_NV_INGEST=${BUILD_NV_INGEST:-1} # 1 = build by default, 0 = skip
BUILD_NV_INGEST_API=${BUILD_NV_INGEST_API:-1} # 1 = build by default, 0 = skip
BUILD_NV_INGEST_CLIENT=${BUILD_NV_INGEST_CLIENT:-1} # 1 = build by default, 0 = skip

##############################
# Package Directories
##############################
NV_INGEST_DIR="${BUILD_SCRIPT_BASE}/packages/nv_ingest"
NV_INGEST_API_DIR="${BUILD_SCRIPT_BASE}/packages/nv_ingest_api"
NV_INGEST_CLIENT_DIR="${BUILD_SCRIPT_BASE}/packages/nv_ingest_client"

##############################
# Setup Output Dir
##############################
echo "Using OUTPUT_DIR: $OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/linux-64"

GIT_SHA=$(git rev-parse --short HEAD)

##############################
# Build Packages
##############################
if [[ "${BUILD_NV_INGEST_API}" -eq 1 ]]; then
    echo "Building nv_ingest_api..."
    SCRIPT_PATH="$GIT_ROOT/api/src/version.py"
    echo "SCRIPT_PATH: $SCRIPT_PATH"
    NV_INGEST_API_VERSION=$(python3 -c "import sys, importlib.util; spec = importlib.util.spec_from_file_location('version', '$SCRIPT_PATH'); version = importlib.util.module_from_spec(spec); spec.loader.exec_module(version); print(version.get_version())")
    echo "NV_INGEST_API_VERSION: $NV_INGEST_API_VERSION"
    NV_INGEST_API_VERSION="${NV_INGEST_API_VERSION}" GIT_ROOT="${GIT_ROOT}/api" GIT_SHA="${GIT_SHA}" conda build "${NV_INGEST_API_DIR}" \
        -c nvidia/label/dev -c rapidsai -c nvidia -c conda-forge -c pytorch \
        --output-folder "${OUTPUT_DIR}" --no-anaconda-upload
else
    echo "Skipping nv_ingest_api build."
fi

if [[ "${BUILD_NV_INGEST}" -eq 1 ]]; then
    echo "Building nv_ingest..."
    GIT_ROOT="${GIT_ROOT}" GIT_SHA="${GIT_SHA}" conda build "${NV_INGEST_DIR}" \
        -c nvidia/label/dev -c rapidsai -c nvidia -c conda-forge -c pytorch \
        --output-folder "${OUTPUT_DIR}" --no-anaconda-upload
else
    echo "Skipping nv_ingest build."
fi

if [[ "${BUILD_NV_INGEST_CLIENT}" -eq 1 ]]; then
    echo "Building nv_ingest_client..."
    SCRIPT_PATH="$GIT_ROOT/client/src/version.py"
    echo "SCRIPT_PATH: $SCRIPT_PATH"
    NV_INGEST_CLIENT_VERSION=$(python3 -c "import sys, importlib.util; spec = importlib.util.spec_from_file_location('version', '$SCRIPT_PATH'); version = importlib.util.module_from_spec(spec); spec.loader.exec_module(version); print(version.get_version())")
    echo "NV_INGEST_CLIENT_VERSION: $NV_INGEST_CLIENT_VERSION"
    NV_INGEST_CLIENT_VERSION="${NV_INGEST_CLIENT_VERSION}" GIT_ROOT="${GIT_ROOT}/client" GIT_SHA="${GIT_SHA}" conda build "${NV_INGEST_CLIENT_DIR}" \
        -c nvidia/label/dev -c rapidsai -c nvidia -c conda-forge -c pytorch \
        --output-folder "${OUTPUT_DIR}" --no-anaconda-upload
else
    echo "Skipping nv_ingest_client build."
fi

##############################
# Index the Conda Channel
##############################
echo "Indexing conda channel at ${OUTPUT_DIR}..."
conda index "${OUTPUT_DIR}"

##############################
# Publish to User-Specified Conda Channel
##############################
publish_to_conda_channel() {
    local channel_path=$1
    echo "Publishing to Conda channel at ${channel_path} (stubbed function)"
    # TODO(Devin): Implement publishing logic (e.g., upload to Anaconda Cloud or other server)
}

if [[ -n "${CONDA_CHANNEL}" ]]; then
    publish_to_conda_channel "${CONDA_CHANNEL}"
else
    echo "No Conda channel specified. Skipping publishing step."
fi

echo "Artifacts successfully built and placed in ${OUTPUT_DIR}"
