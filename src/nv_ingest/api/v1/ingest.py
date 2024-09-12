# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: skip-file

import logging
import traceback
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from redis import RedisError

from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta

logger = logging.getLogger("uvicorn")

router = APIRouter()


async def _get_ingest_service() -> IngestServiceMeta:
    """
    Gather the appropriate Ingestion Service to use for the nv-ingest endpoint.
    """
    logger.debug("Creating RedisIngestService singleton for dependency injection")
    return RedisIngestService.getInstance()


INGEST_SERVICE_T = Annotated[IngestServiceMeta, Depends(_get_ingest_service)]


# POST /submit_job
@router.post(
    "/submit_job",
    responses={
        200: {"description": "Jobs were successfully submitted"},
        500: {"description": "Error encountered while submitting jobs."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="submit jobs to the core nv ingestion service for processing",
    operation_id="submit_job",
)
async def submit_job(job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T):
    try:
        submitted_job_id = await ingest_service.submit_job(job_spec)
        return submitted_job_id
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


# GET /fetch_job
@router.get(
    "/fetch_job/{job_id}",
    responses={
        200: {"description": "Job was successfully retrieved."},
        202: {"description": "Job is not ready yet. Retry later."},
        500: {"description": "Error encountered while fetching job."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="Fetch a previously submitted job from the ingestion service by providing its job_id",
    operation_id="fetch_job",
)
async def fetch_job(job_id: str, ingest_service: INGEST_SERVICE_T):
    try:
        # Attempt to fetch the job from the ingest service
        job_response = await ingest_service.fetch_job(job_id)
        return job_response
    except TimeoutError:
        # Return a 202 Accepted if the job is not ready yet
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")
    except RedisError:
        # Return a 202 Accepted if the job could not be fetched due to Redis error, indicating a retry might succeed
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")
    except ValueError as ve:
        # Return a 500 Internal Server Error for ValueErrors
        raise HTTPException(status_code=500, detail=f"Value error encountered: {str(ve)}")
    except Exception as ex:
        # Catch-all for other exceptions, returning a 500 Internal Server Error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")
