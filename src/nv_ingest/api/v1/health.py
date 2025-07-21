# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

from nv_ingest_api.internal.primitives.nim.model_interface.helpers import is_ready

# logger = logging.getLogger("uvicorn")
logger = logging.getLogger(__name__)

router = APIRouter()

# List of ALL of the HTTP environment variable endpoints that should be checked
READY_CHECK_ENV_VAR_MAP = {
    "ocr": "OCR_HTTP_ENDPOINT",
    "yolox_graphic_elements": "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
    "yolox_page_elements": "YOLOX_HTTP_ENDPOINT",
    "yolox_table_structure": "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
}


@router.get(
    "/live",
    tags=["Health"],
    summary="Check if the service is running.",
    description="""
        Check if the service is running.
    """,
    status_code=status.HTTP_200_OK,
)
async def get_live_state() -> dict:
    live_content = {"live": True}
    return JSONResponse(content=live_content, status_code=200)


@router.get(
    "/ready",
    tags=["Health"],
    summary="Check if the service is ready to receive traffic.",
    description="""
        Check if the service is ready to receive traffic.
    """,
    status_code=status.HTTP_200_OK,
)
async def get_ready_state() -> dict:
    # "Ready" to use means this.
    # 1. nv-ingest FastAPI is live, check you are here nothing to do.
    # 2. Ray pipeline is up and running
    # 3. NIMs that are configured by the service are reporting "ready"
    # After all of those are "ready" this service returns "ready" as well
    # Otherwise a HTTP 503 Service not Available response is returned.

    ingest_ready = True
    # Need to explore options for process checking here.
    # We cannot guarantee this process is local to check.
    # If it is not local and we cannot find a running version
    # locally we could be blocking processing with our
    # readiness endpoint which is really bad. I think it safe
    # for now to assume that if nv-ingest is running so is
    # the pipeline.
    pipeline_ready = True

    # Components that the service should check for "ready"
    # Possible options are
    # 1. empty/none -> This equates to disabling ready checks
    # 2. all/ALL -> This equates to checking the environment variables and checking all configured services for "ready"
    # 3. {comma_delimited_list} -> Comma delimited list of {NIM_HTTP_ENDPOINTS} that should be checked for ready
    components_to_check = os.getenv("COMPONENTS_TO_READY_CHECK", "ALL").upper()

    if components_to_check == "" or components_to_check is None:
        # Ready checks disabled. Immdiately return "ready" status
        return JSONResponse(content={"ready": True}, status_code=200)
    else:
        # Determine the list of components to check, either ALL or a specified list
        endpoint_nim_name_map = {}
        if components_to_check == "ALL":
            # Gather all the known HTTP env endpoints
            for nim_name, nim_env_var in READY_CHECK_ENV_VAR_MAP.items():
                endpoint_url = os.getenv(nim_env_var, None)
                endpoint_nim_name_map[endpoint_url] = nim_name
        else:
            # This will be a list of env variables for the http endpoints to check
            for env_var in components_to_check.split(","):
                env_var = env_var.strip()
                endpoint_url = os.getenv(env_var, None)

                # Get the user friendly name for the NIM from the endpoints map
                nim_name = next((k for k, v in READY_CHECK_ENV_VAR_MAP.items() if v == env_var), None)
                endpoint_nim_name_map[endpoint_url] = nim_name

        # Check the endpoints for their readiness
        ready_statuses = {"ingest_ready": ingest_ready, "pipeline_ready": pipeline_ready}
        ready_to_work = True  # consider nv-ingest ready until an endpoint proves otherwise
        for endpoint, nim_name in endpoint_nim_name_map.items():
            endpoint_ready = is_ready(endpoint, "/v1/health/ready")
            if not endpoint_ready:
                logger.debug(f"Not ready for work. NIM endpoint: '{endpoint}' reporting not ready.")
                ready_to_work = False
                ready_statuses[nim_name + "_ready"] = False
            else:
                ready_statuses[nim_name + "_ready"] = True

        # Build the response for the client
        if ready_to_work:
            return JSONResponse(content={"ready": True}, status_code=200)
        else:
            logger.debug(f"Ready Statuses: {ready_statuses}")
            return JSONResponse(content=ready_statuses, status_code=503)
