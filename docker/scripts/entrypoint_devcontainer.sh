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

# Source "source" file if it exists
SRC_FILE="/opt/docker/bin/entrypoint_source"
[ -f "${SRC_FILE}" ] && source "${SRC_FILE}"

# Check if user supplied a command
if [ "$#" -gt 0 ]; then
    # If a command is provided, run it
    exec "$@"
else
    # If no command is provided, run the default startup launch
    if [ "${MESSAGE_CLIENT_TYPE}" != "simple" ]; then
      # Start uvicorn if MESSAGE_CLIENT_TYPE is not 'simple'
      uvicorn nv_ingest.api.main:app --workers 1 --host 0.0.0.0 --port 7670 --reload --app-dir /workspace/src/nv_ingest &
    fi

    python /workspace/src/microservice_entrypoint.py
fi
