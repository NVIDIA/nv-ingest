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

from client.src.nv_ingest_client.primitives.jobs.job_spec import JobSpec
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta

# from nv_ingest_client.primitives.jobs.job_spec import JobSpec
# from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
# from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta

logger = logging.getLogger(__name__)

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
async def submit_job(job_spec_json: dict, ingest_service: INGEST_SERVICE_T):
    try:
        job_spec = JobSpec.from_dict(job_spec_json)
        submitted_job_id = await ingest_service.submit_job(job_spec)
        print(f"Submitted Job_Id: {submitted_job_id}")
        return submitted_job_id
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


# POST /fetch_job
@router.get(
    "/fetch_job/{job_id}",
    responses={
        200: {"description": "Job was succesfully retrieved."},
        500: {"description": "Error encountered while fetching job."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="Fetch a previously submitted job from the ingestion service by providing its job_id",
    operation_id="fetch_job",
)
async def fetch_job(job_id: str, ingest_service: INGEST_SERVICE_T):
    logger.info(f"!!!! Entering fetch_job endpoint: {job_id}")
    print(f"!!!! Entering fetch_job endpoint: {job_id}")
    try:
        job_response = await ingest_service.fetch_job(job_id)
        return job_response
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")
