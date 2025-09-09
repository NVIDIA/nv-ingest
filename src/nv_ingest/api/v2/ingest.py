# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from opentelemetry import trace
from pydantic import ValidationError

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.util.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from .schemas import SubmitJobRequestV2, SubmitJobResponseV2

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/v2", tags=["Ingestion V2"])


async def _get_ingest_service() -> IngestServiceMeta:
    """Dependency injection for ingest service"""
    return RedisIngestService.get_instance()


INGEST_SERVICE_T = Annotated[IngestServiceMeta, Depends(_get_ingest_service)]


def trace_id_to_uuid(trace_id: str) -> str:
    """Convert a 32-character OpenTelemetry trace ID to a UUID-like format."""
    trace_id = str(trace.format_trace_id(trace_id))
    if len(trace_id) != 32:
        raise ValueError("Trace ID must be a 32-character hexadecimal string")
    return f"{trace_id[:8]}-{trace_id[8:12]}-{trace_id[12:16]}-{trace_id[16:20]}-{trace_id[20:]}"


@router.post(
    "/submit_job",
    response_model=SubmitJobResponseV2,
    responses={
        200: {"description": "Job successfully submitted with type safety"},
        400: {"description": "Invalid request data - validation failed"},
        500: {"description": "Internal server error during submission"},
        503: {"description": "Service unavailable"},
    },
    summary="Submit job with type-safe validation (V2 API)",
    operation_id="submit_job_v2",
)
async def submit_job_v2(
    request: Request, response: Response, job_request: SubmitJobRequestV2, ingest_service: INGEST_SERVICE_T
) -> SubmitJobResponseV2:
    """
    V2 job submission endpoint with type-safe request handling.

    Replaces raw JSON manipulation from v1 with Pydantic model validation.
    Provides better error messages and eliminates runtime dictionary access errors.
    """
    with tracer.start_as_current_span("http-submit-job-v2") as span:
        try:
            # Add telemetry attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("api.version", "v2")
            span.add_event("Processing V2 typed job submission")

            current_trace_id = span.get_span_context().trace_id
            job_id = trace_id_to_uuid(current_trace_id)

            # TYPE-SAFE conversion (replaces raw JSON manipulation)
            try:
                job_spec_dict = job_request.to_job_spec_with_tracing(str(current_trace_id))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in job_spec_json: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON in job_spec_json: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing job request: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing job request: {str(e)}")

            # Bridge to existing service infrastructure
            updated_job_spec = MessageWrapper(payload=json.dumps(job_spec_dict))

            span.add_event("Submitting to service layer")

            # Use existing service layer (no changes needed)
            await ingest_service.submit_job(updated_job_spec, job_id)
            await ingest_service.set_job_state(job_id, "SUBMITTED")

            # Return typed response
            response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
            span.add_event("V2 job submission completed successfully")

            return SubmitJobResponseV2(job_id=job_id, trace_id=trace.format_trace_id(current_trace_id))

        except ValidationError as ve:
            # Pydantic validation errors (should be caught by FastAPI, but just in case)
            logger.warning(f"Validation error in V2 submission: {ve}")
            raise HTTPException(status_code=400, detail=f"Request validation failed: {str(ve)}")

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise

        except Exception as ex:
            logger.exception(f"Unexpected error in V2 job submission: {str(ex)}")
            raise HTTPException(status_code=500, detail=f"V2 API Internal Server Error: {str(ex)}")
