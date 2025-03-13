# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

from nv_ingest.util.nim.helpers import is_ready

logger = logging.getLogger("uvicorn")

router = APIRouter()


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
    # 2. Morpheus pipeline is up and running
    # 3. Yolox NIM "ready" health endpoint returns 200 status code
    # 4. Deplot NIM "ready" health endpoint returns 200 status code
    # 5. Cached NIM "ready" health endpoint returns 200 status code
    # 6. PaddleOCR NIM "ready" health endpoint returns 200 status code
    # 7. Embedding NIM "ready" health endpoint returns 200 status code
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
    morpheus_pipeline_ready = True

    # We give the users an option to disable checking all distributed services for "readiness"
    check_all_components = os.getenv("READY_CHECK_ALL_COMPONENTS", "True").lower()
    if check_all_components in ["1", "true", "yes"]:
        yolox_page_elements_ready = is_ready(os.getenv("YOLOX_HTTP_ENDPOINT", None), "/v1/health/ready")
        yolox_graphic_elements_ready = is_ready(
            os.getenv("YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT", None), "/v1/health/ready"
        )
        paddle_ready = is_ready(os.getenv("PADDLE_HTTP_ENDPOINT", None), "/v1/health/ready")

        if (
            ingest_ready
            and morpheus_pipeline_ready
            and yolox_page_elements_ready
            and yolox_graphic_elements_ready
            and paddle_ready
        ):
            return JSONResponse(content={"ready": True}, status_code=200)
        else:
            ready_statuses = {
                "ingest_ready": ingest_ready,
                "morpheus_pipeline_ready": morpheus_pipeline_ready,
                "page_elemenst_ready": yolox_page_elements_ready,
                "graphic_elements_ready": yolox_graphic_elements_ready,
                "paddle_ready": paddle_ready,
            }
            logger.debug(f"Ready Statuses: {ready_statuses}")
            return JSONResponse(content=ready_statuses, status_code=503)
    else:
        return JSONResponse(content={"ready": True}, status_code=200)
