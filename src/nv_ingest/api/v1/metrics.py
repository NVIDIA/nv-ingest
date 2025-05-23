# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging

from fastapi import APIRouter, Response, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()

# logger = logging.getLogger("uvicorn")
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["method", "endpoint"])


@router.get(
    "/metrics",
    tags=["Health"],
    summary="Provide prometheus formatted metrics for consumption",
    description="""
        Provide prometheus formatted metrics for consumption by a prometheus scraping server.
    """,
    status_code=status.HTTP_200_OK,
)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
