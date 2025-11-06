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


#!/bin/bash

set -e

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && . "${SRC_FILE}"

SRC_EXT="/workspace/docker/entrypoint_source_ext.sh"
[ -f "${SRC_EXT}" ] && . "${SRC_EXT}"

# Determine ingest config path, if exists and is a valid file.
if [ -n "${INGEST_CONFIG_PATH}" ] && [ -f "${INGEST_CONFIG_PATH}" ]; then
    CONFIG_ARG="--pipeline-config-path=${INGEST_CONFIG_PATH}"
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
    # Normalize log level: default to 'info' if INGEST_LOG_LEVEL is 'DEFAULT'
    _log_level=$(echo "${INGEST_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')
    if [ "$_log_level" = "default" ]; then
        _log_level="warning"
    fi

    # --- Determine if access logs should be enabled ---
    _access_logs_enabled="false"

    # Normalize INGEST_ENABLE_SERVICE_ACCESS_LOGS to lower case
    _explicit_access_logs=$(echo "${INGEST_ENABLE_SERVICE_ACCESS_LOGS:-false}" | tr '[:upper:]' '[:lower:]')

    if [ "$_log_level" = "debug" ]; then
        _access_logs_enabled="true"
    elif [ "$_explicit_access_logs" = "true" ] || [ "$_explicit_access_logs" = "1" ] || [ "$_explicit_access_logs" = "yes" ]; then
        _access_logs_enabled="true"
    fi

    # Set gunicorn access log settings
    if [ "$_access_logs_enabled" = "true" ]; then
        _gunicorn_access_logfile="-"
        _gunicorn_access_logformat='default'
    else
        _gunicorn_access_logfile="/dev/null"
        _gunicorn_access_logformat=''
    fi

    # --- Launch Services ---

    if [ "${MESSAGE_CLIENT_TYPE}" != "simple" ]; then
        # Start gunicorn if MESSAGE_CLIENT_TYPE is not 'simple'.
        gunicorn nv_ingest.api.main:app \
            -w 32 \
            -k uvicorn.workers.UvicornWorker \
            --bind 0.0.0.0:7670 \
            --timeout 300 \
            --log-level "${_log_level}" \
            --access-logfile "${_gunicorn_access_logfile}" \
            --access-logformat "${_gunicorn_access_logformat}" \
            --error-logfile - &
    fi

    if [ "${MEM_TRACE}" = true ]; then
        # Run the entrypoint wrapped in memray
        python -m memray run -o memray_trace.bin /workspace/microservice_entrypoint.py ${CONFIG_ARG}
    else
        # Run without memray
        python /workspace/microservice_entrypoint.py ${CONFIG_ARG}
    fi
fi
