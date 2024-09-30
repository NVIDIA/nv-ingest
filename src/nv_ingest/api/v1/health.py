# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from opentelemetry import trace
from fastapi import APIRouter
from fastapi import status
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()


@router.get(
    "/health/live",
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
    "/health/ready",
    tags=["Health"],
    summary="Check if the service is ready to receive traffic.",
    description="""
        Check if the service is ready to receive traffic.
    """,
    status_code=status.HTTP_200_OK,
)
async def get_ready_state() -> dict:
    ready_content = {"ready": True}
    return JSONResponse(content=ready_content, status_code=200)
