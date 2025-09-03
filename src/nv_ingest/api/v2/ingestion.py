# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

from io import BytesIO
from typing import Annotated, Dict, List
import base64
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi import Depends
from fastapi import File, UploadFile, Form
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.schemas.framework_processing_job_schema import ProcessingJob, ConversionStatus
from nv_ingest.framework.util.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks.extract import ExtractTask
from opentelemetry import trace
from redis import RedisError

from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask
from nv_ingest_client.primitives.tasks.infographic_extraction import InfographicExtractionTask

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()


from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class IngestionProcessingOptions(BaseModel):
    extract_text: Optional[bool] = Field(default=True, description="Whether to extract text from documents")
    extract_images: Optional[bool] = Field(default=False, description="Whether to extract images from documents")
    extract_tables: Optional[bool] = Field(default=False, description="Whether to extract tables from documents")
    custom_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata for processing")


class IngestionJobRequest(BaseModel):
    options: IngestionProcessingOptions = Field(..., description="Processing options for the uploaded files")
    description: Optional[str] = Field(default=None, description="Description of the ingestion job")


class IngestionJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the ingestion job")
    status: str = Field(..., description="Status of the ingestion job")
    submitted_files: List[str] = Field(..., description="List of filenames submitted")
    options: IngestionProcessingOptions = Field(..., description="Processing options used for the job")
    description: Optional[str] = Field(default=None, description="Description of the ingestion job")


# In-memory store for ingestion jobs (for demonstration; replace with persistent store in production)
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/ingestion",
    response_model=IngestionJobResponse,
    tags=["Ingestion"],
    summary="Submit one or more documents for ingestion with processing options",
    status_code=201,
)
async def create_ingestion_job(
    files: List[UploadFile] = File(..., description="One or more files to ingest"),
    options: Optional[str] = Form(None, description="Optional JSON string of processing options and job description"),
):
    """
    Accepts multipart form uploads of documents and an optional JSON string describing processing options.
    """
    import time
    import uuid
    import json

    if options is not None:
        try:
            options_dict = json.loads(options)
            ingestion_request = IngestionJobRequest(**options_dict)
        except Exception as ex:
            logger.exception("Invalid options JSON provided.")
            raise HTTPException(status_code=400, detail=f"Invalid options JSON: {str(ex)}")
    else:
        # Use default IngestionJobRequest (with default options and no description)
        ingestion_request = IngestionJobRequest(options=IngestionProcessingOptions(), description=None)

    job_id = str(uuid.uuid4())
    submitted_files = []
    for file in files:
        # In a real implementation, save file to storage and queue for processing
        submitted_files.append(file.filename)

        file.file.seek(0, 2)  # Move to end of file
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to start

        if file_size <= 1024 * 1024:
            logger.info("Small file. Priority QoS.")
        else:
            logger.info("Large file. Slower QoS")
            time.sleep(30)

        # file.file.read()  # Read file content if needed

    job_record = {
        "job_id": job_id,
        "status": "SUBMITTED",
        "submitted_files": submitted_files,
        "options": ingestion_request.options.dict(),
        "description": ingestion_request.description,
        "created_at": time.time(),
    }
    INGESTION_JOBS[job_id] = job_record

    return IngestionJobResponse(
        job_id=job_id,
        status="SUBMITTED",
        submitted_files=submitted_files,
        options=ingestion_request.options,
        description=ingestion_request.description,
    )


@router.get(
    "/ingestion",
    response_model=List[IngestionJobResponse],
    tags=["Ingestion"],
    summary="Get all ingestion jobs",
    status_code=200,
)
async def get_all_ingestion_jobs():
    """
    Returns all ingestion jobs.
    """
    return [
        IngestionJobResponse(
            job_id=job["job_id"],
            status=job["status"],
            submitted_files=job["submitted_files"],
            options=IngestionProcessingOptions(**job["options"]),
            description=job.get("description"),
        )
        for job in INGESTION_JOBS.values()
    ]


@router.get(
    "/ingestion/{job_id}",
    response_model=IngestionJobResponse,
    tags=["Ingestion"],
    summary="Get a specific ingestion job by job_id",
    status_code=200,
)
async def get_ingestion_job(job_id: str):
    """
    Returns the details of a specific ingestion job.
    """
    job = INGESTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return IngestionJobResponse(
        job_id=job["job_id"],
        status=job["status"],
        submitted_files=job["submitted_files"],
        options=IngestionProcessingOptions(**job["options"]),
        description=job.get("description"),
    )


@router.put(
    "/ingestion/{job_id}",
    response_model=IngestionJobResponse,
    tags=["Ingestion"],
    summary="Update an existing ingestion job's processing options or description",
    status_code=200,
)
async def update_ingestion_job(
    job_id: str,
    update: IngestionJobRequest,
):
    """
    Updates the processing options or description for an existing ingestion job.
    """
    job = INGESTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    job["options"] = update.options.dict()
    job["description"] = update.description
    return IngestionJobResponse(
        job_id=job["job_id"],
        status=job["status"],
        submitted_files=job["submitted_files"],
        options=update.options,
        description=update.description,
    )


@router.delete(
    "/ingestion/{job_id}",
    response_model=IngestionJobResponse,
    tags=["Ingestion"],
    summary="Delete an ingestion job",
    status_code=200,
)
async def delete_ingestion_job(job_id: str):
    """
    Deletes an ingestion job.
    """
    job = INGESTION_JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return IngestionJobResponse(
        job_id=job["job_id"],
        status=job["status"],
        submitted_files=job["submitted_files"],
        options=IngestionProcessingOptions(**job["options"]),
        description=job.get("description"),
    )
