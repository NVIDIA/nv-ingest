#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Activate the `nv_ingest_runtime` conda environment
. /opt/conda/etc/profile.d/conda.sh
conda activate nv_ingest_runtime

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && . "${SRC_FILE}"

SRC_EXT="/workspace/docker/entrypoint_source_ext.sh"
[ -f "${SRC_EXT}" ] && . "${SRC_EXT}"

# Determine edge buffer size (default: 32)
EDGE_BUFFER_SIZE="${INGEST_EDGE_BUFFER_SIZE:-32}"

# Determine ingest config path, if exists and is a valid file.
if [ -n "${INGEST_CONFIG_PATH}" ] && [ -f "${INGEST_CONFIG_PATH}" ]; then
    CONFIG_ARG="--ingest_config_path=${INGEST_CONFIG_PATH}"
else
    CONFIG_ARG=""
fi

# Check if INGEST_MEM_TRACE is set to 1, true, on, or yes (case-insensitive)
MEM_TRACE=false
if [ -n "${INGEST_MEM_TRACE}" ]; then
    case "$(echo "${INGEST_MEM_TRACE}" | tr '[:upper:]' '[:lower:]')" in
        1|true|on|yes)
            MEM_TRACE=true
            ;;
    esac
fi

# Check if user supplied a command
if [ "$#" -gt 0 ]; then
    # If a command is provided, run it.
    exec "$@"
else
    # If no command is provided, run the default startup launch.
    if [ "${MESSAGE_CLIENT_TYPE}" != "simple" ]; then
        # Start uvicorn if MESSAGE_CLIENT_TYPE is not 'simple'.
        uvicorn nv_ingest.api.main:app --workers 32 --host 0.0.0.0 --port 7670 &
    fi

    if [ "${MEM_TRACE}" = true ]; then
        # Run the entrypoint wrapped in memray
        python -m memray run -o memray_trace.bin /workspace/microservice_entrypoint.py --edge_buffer_size="${EDGE_BUFFER_SIZE}" ${CONFIG_ARG}
    else
        # Run without memray
        python /workspace/microservice_entrypoint.py --edge_buffer_size="${EDGE_BUFFER_SIZE}" ${CONFIG_ARG}
    fi
fi
