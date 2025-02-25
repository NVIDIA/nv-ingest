#!/bin/bash --login
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


#!/bin/bash

# Activate the `nv_ingest_runtime` conda environment
. /opt/conda/etc/profile.d/conda.sh
conda activate nv_ingest_runtime

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && source "${SRC_FILE}"

# Check if user supplied a command
if [ "$#" -gt 0 ]; then
    # If a command is provided, run it.
    exec "$@"
else
    # If no command is provided, run the default startup launch.
    if [ "${MESSAGE_CLIENT_TYPE}" != "simple" ]; then
        # Determine the log level for uvicorn.
        log_level=$(echo "${INGEST_LOG_LEVEL:-default}" | tr '[:upper:]' '[:lower:]')
        if [ "$log_level" = "default" ]; then
            log_level="info"
        fi

        # Build the uvicorn command with the specified log level.
        uvicorn_cmd="uvicorn nv_ingest.main:app --workers 32 --host 0.0.0.0 --port 7670 --log-level ${log_level}"

        # If DISABLE_FAST_API_ACCESS_LOGGING is true, disable access logs.
        if [ "${DISABLE_FAST_API_ACCESS_LOGGING}" == "true" ]; then
            uvicorn_cmd="${uvicorn_cmd} --no-access-log"
        fi

        # Start uvicorn in the background.
        $uvicorn_cmd &
    fi

    # Start the microservice entrypoint.
    python /workspace/microservice_entrypoint.py
fi
