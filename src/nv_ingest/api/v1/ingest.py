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

import base64
import json
import logging
import time
import traceback
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Request, Response
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks.extract import ExtractTask
from opentelemetry import trace
from redis import RedisError

from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()


async def _get_ingest_service() -> IngestServiceMeta:
    """
    Gather the appropriate Ingestion Service to use for the nv-ingest endpoint.
    """
    logger.debug("Creating RedisIngestService singleton for dependency injection")
    return RedisIngestService.getInstance()


INGEST_SERVICE_T = Annotated[IngestServiceMeta, Depends(_get_ingest_service)]


# POST /submit
@router.post(
    "/submit",
    responses={
        200: {"description": "Submission was successful"},
        500: {"description": "Error encountered during submission"},
    },
    tags=["Ingestion"],
    summary="submit document to the core nv ingestion service for processing",
    operation_id="submit",
)
async def submit_job_curl_friendly(ingest_service: INGEST_SERVICE_T, file: UploadFile = File(...)):
    """
    A multipart/form-data friendly Job submission endpoint that makes interacting with
    the nv-ingest service through tools like Curl easier.
    """
    try:
        file_stream = BytesIO(file.file.read())
        doc_content = base64.b64encode(file_stream.read()).decode("utf-8")

        # Construct the JobSpec from the HTTP supplied form-data
        job_spec = JobSpec(
            # TOOD: Update this to look at the uploaded content-type, currently that is not working
            document_type="pdf",
            payload=doc_content,
            source_id=file.filename,
            source_name=file.filename,
            # TODO: Update this to accept user defined options
            extended_options={
                "tracing_options": {
                    "trace": True,
                    "ts_send": time.time_ns(),
                    "trace_id": str(trace.get_current_span().get_span_context().trace_id),
                }
            },
        )

        # This is the "easy submission path" just default to extracting everything
        extract_task = ExtractTask(document_type="pdf", extract_text=True, extract_images=True, extract_tables=True)

        job_spec.add_task(extract_task)

        submitted_job_id = await ingest_service.submit_job(MessageWrapper(payload=json.dumps(job_spec.to_dict())))
        return submitted_job_id
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


def trace_id_to_uuid(trace_id: str) -> str:
    """Convert a 32-character OpenTelemetry trace ID to a UUID-like format."""
    trace_id = str(trace.format_trace_id(trace_id))
    if len(trace_id) != 32:
        raise ValueError("Trace ID must be a 32-character hexadecimal string")
    return f"{trace_id[:8]}-{trace_id[8:12]}-{trace_id[12:16]}-{trace_id[16:20]}-{trace_id[20:]}"


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
async def submit_job(request: Request, response: Response, job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T):
    with tracer.start_as_current_span("http-submit-job") as span:
        try:
            # Add custom attributes to the span
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.add_event("Submitting file for processing")

            # Inject the x-trace-id into the JobSpec definition so that OpenTelemetry
            # will be able to trace across uvicorn -> morpheus
            current_trace_id = span.get_span_context().trace_id

            job_spec_dict = json.loads(job_spec.payload)
            job_spec_dict["tracing_options"]["trace_id"] = str(current_trace_id)
            updated_job_spec = MessageWrapper(payload=json.dumps(job_spec_dict))

            job_id = trace_id_to_uuid(current_trace_id)

            # Submit the job async
            await ingest_service.submit_job(updated_job_spec, job_id)

            # Add another event
            span.add_event("Finished processing")

            # We return the trace-id as a 32-byte hexidecimal string which is the format you would use when
            # searching in Zipkin for traces. The original value is a 128 bit integer ...
            response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)

            return job_id

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
