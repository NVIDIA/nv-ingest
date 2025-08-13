# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from fastapi import Request, HTTPException
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
# The only place that knows the concrete class:
from nv_ingest.framework.util.service.impl.ingest.redis_ingest_service import RedisIngestService

logger = logging.getLogger(__name__)

def create_ingest_service() -> IngestServiceMeta:
    """
    Gather the appropriate Ingestion Service to use for the nv-ingest endpoint.
    Construct the concrete ingest service (Redis today).
    Keeping this import/choice here isolates main.py and routers from the impl.
    In the future, you can swap this to read env or return a different impl.
    """
    
    logger.debug("Creating RedisIngestService singleton for dependency injection")
    return RedisIngestService.get_instance()

async def get_ingest_service(request: Request) -> IngestServiceMeta:
    """
    FastAPI dependency used by routers. Prefer the lifespan instance on app.state.
    As a safe fallback (e.g., unit tests without app startup), lazily create one.
    """
    svc = getattr(request.app.state, "ingest_service", None)
    if svc is None:
        # Fallback keeps local tests/dev working without lifespan.
        # If you prefer to enforce startup, raise 503 instead of creating.
        logger.warning("ingest_service not found on app.state; creating a local instance")
        svc = create_ingest_service()
        # do NOT stash it on app.state here to avoid surprising test sharing
    return svc
