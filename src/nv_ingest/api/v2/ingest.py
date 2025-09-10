# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NV-Ingest V2 API Endpoints

Type-safe ingestion API with comprehensive validation and structured responses.
Eliminates raw JSON manipulation while maintaining full compatibility with
existing service infrastructure.
"""

import json
import logging
import time
from typing import Annotated, List

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from opentelemetry import trace
from pydantic import ValidationError

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.schemas.framework_processing_job_schema import ProcessingJob, ConversionStatus
from nv_ingest.framework.util.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from redis import RedisError
from .schemas import SubmitJobRequestV2, SubmitJobResponseV2, FetchJobResponseV2
from .pdf_splitter import split_pdf_job

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/v2", tags=["Ingestion V2"])

# Job state constants (same as V1)
STATE_RETRIEVED_DESTRUCTIVE = "RETRIEVED_DESTRUCTIVE"
STATE_RETRIEVED_NON_DESTRUCTIVE = "RETRIEVED_NON_DESTRUCTIVE"
STATE_RETRIEVED_CACHED = "RETRIEVED_CACHED"
STATE_FAILED = "FAILED"
STATE_PROCESSING = "PROCESSING"
STATE_SUBMITTED = "SUBMITTED"
INTERMEDIATE_STATES = {STATE_PROCESSING, STATE_SUBMITTED}


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


def _dict_to_job_spec(job_spec_dict: dict) -> JobSpec:
    """Convert job specification dictionary to JobSpec object for PDF splitting."""
    # V2 API uses IngestJobSchema format with job_payload structure
    job_payload = job_spec_dict.get("job_payload", {})

    # Extract content from V2 API structure
    if job_payload:
        content_list = job_payload.get("content", [""])
        document_type_list = job_payload.get("document_type", ["pdf"])
        source_name_list = job_payload.get("source_name", [""])
        source_id_list = job_payload.get("source_id", source_name_list)

        # Take first item from lists
        payload = content_list[0] if content_list else ""
        document_type = document_type_list[0] if document_type_list else "pdf"
        source_name = source_name_list[0] if source_name_list else ""
        source_id = source_id_list[0] if source_id_list else source_name
    else:
        # Fallback to old structure
        payload = job_spec_dict.get("payload", "")
        document_type = job_spec_dict.get("document_type", "pdf")
        source_name = job_spec_dict.get("source_name", "")
        source_id = job_spec_dict.get("source_id", "")

    return JobSpec(
        document_type=document_type,
        payload=payload,
        source_id=source_id,
        source_name=source_name,
        extended_options=job_spec_dict.get("extended_options", {}),
    )


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
    Submit job for processing with comprehensive validation.

    Validates job specification using production schemas before submission.
    Provides detailed error messages for invalid requests.
    Includes PDF splitting for performance optimization.
    """
    with tracer.start_as_current_span("http-submit-job-v2") as span:
        try:
            # Add telemetry attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("api.version", "v2")
            span.add_event("Processing job submission")

            current_trace_id = span.get_span_context().trace_id
            job_id = trace_id_to_uuid(current_trace_id)

            # Validate and prepare job specification
            try:
                job_spec_dict = job_request.to_job_spec_with_tracing(str(current_trace_id))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid JSON in job request: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON in job_spec_json: {str(e)}")
            except ValidationError as ve:
                logger.warning(f"Job specification validation failed: {ve}")
                raise HTTPException(status_code=400, detail=f"Job spec validation failed: {str(ve)}")
            except Exception as e:
                logger.error(f"Error processing job request: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing job request: {str(e)}")

            # ======================================================================
            # PDF SPLITTING LOGIC
            # ======================================================================
            try:
                # Convert to JobSpec for PDF splitting analysis
                job_spec_for_splitting = _dict_to_job_spec(job_spec_dict)

                # Analyze and potentially split the PDF
                span.add_event("Analyzing PDF for splitting")
                split_result = split_pdf_job(job_spec_for_splitting, job_id)

                if split_result.should_split:
                    logger.info(f"Splitting PDF job {job_id} into {len(split_result.subjobs)} subjobs")

                    # Enhanced telemetry for PDF splitting
                    span.set_attribute("pdf.split", True)
                    span.set_attribute("pdf.total_pages", split_result.total_pages)
                    span.set_attribute("pdf.subjob_count", len(split_result.subjobs))
                    span.set_attribute("pdf.pages_per_subjob", 3)  # Our configured value
                    span.set_attribute("pdf.threshold_pages", 4)  # Our configured threshold

                    # Add detailed events for observability
                    span.add_event(
                        "PDF splitting analysis completed",
                        {
                            "pages": split_result.total_pages,
                            "subjobs_to_create": len(split_result.subjobs),
                            "splitting_method": "pypdfium2",
                        },
                    )

                    # Create parent job metadata using processing cache
                    parent_jobs: List[ProcessingJob] = []
                    for subjob_data in split_result.subjobs:
                        parent_jobs.append(
                            ProcessingJob(
                                submitted_job_id=subjob_data["subjob_id"],
                                filename=subjob_data["subjob_id"],
                                status=ConversionStatus.IN_PROGRESS,
                            )
                        )

                    # Store parent job metadata
                    await ingest_service.set_processing_cache(job_id, parent_jobs)

                    # Submit all subjobs with enhanced tracing
                    span.add_event("Submitting PDF subjobs to processing pipeline")

                    subjob_submission_start = time.time()
                    submitted_subjobs = []

                    for subjob_data in split_result.subjobs:
                        subjob_id = subjob_data["subjob_id"]
                        subjob_spec = subjob_data["job_spec"]

                        # Create child span for each subjob submission
                        with tracer.start_as_current_span("submit-subjob") as subjob_span:
                            subjob_span.set_attribute("subjob.id", subjob_id)
                            subjob_span.set_attribute("subjob.parent_id", job_id)
                            subjob_span.set_attribute("subjob.pages", subjob_data.get("pages", 3))

                            # Convert subjob to dict with tracing
                            subjob_dict = subjob_spec.to_dict()
                            subjob_dict["tracing_options"] = job_spec_dict.get("tracing_options", {})

                            subjob_message = MessageWrapper(payload=json.dumps(subjob_dict))
                            await ingest_service.submit_job(subjob_message, subjob_id)
                            await ingest_service.set_job_state(subjob_id, "SUBMITTED")

                            submitted_subjobs.append(subjob_id)
                            subjob_span.add_event("Subjob submitted to pipeline")
                            logger.debug(f"Submitted subjob {subjob_id}")

                    subjob_submission_time = time.time() - subjob_submission_start

                    # Set parent job state
                    await ingest_service.set_job_state(job_id, "SUBMITTED_SPLIT")

                    # Enhanced completion telemetry
                    span.add_event(
                        "PDF splitting completed",
                        {
                            "subjobs_submitted": len(submitted_subjobs),
                            "submission_time_ms": round(subjob_submission_time * 1000, 2),
                            "avg_submission_time_ms": round(
                                (subjob_submission_time / len(submitted_subjobs)) * 1000, 2
                            ),
                        },
                    )
                    span.set_attribute("pdf.submission_time_ms", round(subjob_submission_time * 1000, 2))

                else:
                    # No splitting needed - proceed with normal submission
                    span.set_attribute("pdf.split", False)
                    span.add_event("PDF below splitting threshold - submitting as single job")

                    job_message = MessageWrapper(payload=json.dumps(job_spec_dict))
                    await ingest_service.submit_job(job_message, job_id)
                    await ingest_service.set_job_state(job_id, "SUBMITTED")

            except Exception as split_error:
                logger.warning(f"PDF splitting failed for job {job_id}: {split_error}")
                span.add_event(f"PDF splitting failed: {split_error}")

                # Fallback to normal submission without splitting
                logger.info(f"Falling back to single job submission for {job_id}")
                job_message = MessageWrapper(payload=json.dumps(job_spec_dict))
                await ingest_service.submit_job(job_message, job_id)
                await ingest_service.set_job_state(job_id, "SUBMITTED")

            # Return success response
            response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
            span.add_event("Job submission completed successfully")

            return SubmitJobResponseV2(job_id=job_id, trace_id=trace.format_trace_id(current_trace_id))

        except ValidationError as ve:
            logger.warning(f"Request validation error: {ve}")
            raise HTTPException(status_code=400, detail=f"Request validation failed: {str(ve)}")

        except HTTPException:
            raise

        except Exception as ex:
            logger.exception(f"Unexpected error in job submission: {str(ex)}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(ex)}")


@router.get(
    "/fetch_job/{job_id}",
    response_model=FetchJobResponseV2,
    responses={
        200: {"description": "Job result successfully retrieved"},
        202: {"description": "Job is processing or result not yet available. Retry later"},
        404: {"description": "Job ID not found or associated state has expired"},
        410: {"description": "Job result existed but is now gone (expired or retrieved destructively/cached)"},
        500: {"description": "Internal server error during fetch processing"},
        503: {"description": "Job processing failed, or backend service temporarily unavailable preventing fetch"},
    },
    summary="Fetch job result with structured response",
    operation_id="fetch_job_v2",
)
async def fetch_job_v2(job_id: str, ingest_service: INGEST_SERVICE_T) -> FetchJobResponseV2:
    """
    Fetch job result with comprehensive state management.

    Returns structured response data with proper error handling for all job states.
    Distinguishes between non-existent jobs (404) and expired results (410).
    """
    with tracer.start_as_current_span("http-fetch-job-v2") as span:
        try:
            # Add telemetry attributes
            span.set_attribute("api.version", "v2")
            span.set_attribute("job.id", job_id)
            span.add_event("Processing job fetch request")

            # Check job state
            current_state = await ingest_service.get_job_state(job_id)
            logger.debug(f"Job {job_id} state: {current_state}")

            if current_state is None:
                logger.warning(f"Job {job_id} not found or expired")
                raise HTTPException(status_code=404, detail="Job ID not found or state has expired")

            if current_state == STATE_FAILED:
                logger.error(f"Job {job_id} processing failed")
                raise HTTPException(status_code=503, detail="Job processing failed")

            if current_state == STATE_RETRIEVED_DESTRUCTIVE:
                logger.warning(f"Job {job_id} was destructively retrieved")
                raise HTTPException(status_code=410, detail="Job result is gone (destructive read)")

            # Attempt fetch for valid states
            if current_state in INTERMEDIATE_STATES or current_state in {
                STATE_RETRIEVED_NON_DESTRUCTIVE,
                STATE_RETRIEVED_CACHED,
            }:
                logger.debug(f"Fetching job {job_id} in state {current_state}")

                try:
                    # Fetch job data from service
                    job_response = await ingest_service.fetch_job(job_id)
                    logger.debug(f"Successfully fetched job {job_id}")

                    # Update job state based on fetch mode
                    try:
                        current_fetch_mode = await ingest_service.get_fetch_mode()
                        if current_fetch_mode == FetchMode.DESTRUCTIVE:
                            target_state = STATE_RETRIEVED_DESTRUCTIVE
                        elif current_fetch_mode == FetchMode.NON_DESTRUCTIVE:
                            target_state = STATE_RETRIEVED_NON_DESTRUCTIVE
                        elif current_fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
                            target_state = STATE_RETRIEVED_CACHED
                        else:
                            target_state = "RETRIEVED_UNKNOWN"

                        if target_state != "RETRIEVED_UNKNOWN":
                            await ingest_service.set_job_state(job_id, target_state)
                            logger.debug(f"Updated job {job_id} state to {target_state}")

                    except Exception as state_err:
                        logger.error(f"Failed to update job state for {job_id}: {state_err}")

                    # Create structured response
                    try:
                        span.add_event("Creating structured response")

                        # Ensure response data is properly formatted
                        if isinstance(job_response, (dict, list)):
                            response_data = job_response
                        else:
                            response_data = json.loads(str(job_response))

                        return FetchJobResponseV2(
                            job_id=job_id,
                            status="completed",
                            results=response_data if isinstance(response_data, list) else [response_data],
                            trace_id=trace.format_trace_id(span.get_span_context().trace_id),
                        )

                    except (TypeError, ValueError, json.JSONDecodeError) as json_err:
                        logger.exception(f"Error serializing response for job {job_id}: {json_err}")
                        raise HTTPException(status_code=500, detail="Internal server error: Failed to serialize result")

                except (TimeoutError, RedisError, ConnectionError) as fetch_err:
                    fetch_err_type = type(fetch_err).__name__

                    if isinstance(fetch_err, TimeoutError):
                        logger.info(f"Job {job_id} still processing, fetch timed out")
                    else:
                        logger.warning(f"Backend error ({fetch_err_type}) fetching job {job_id}: {fetch_err}")

                    # Handle errors based on job state
                    if current_state == STATE_RETRIEVED_NON_DESTRUCTIVE:
                        if isinstance(fetch_err, TimeoutError):
                            raise HTTPException(status_code=410, detail="Job result is gone (TTL expired)")
                        else:
                            raise HTTPException(
                                status_code=503, detail="Backend service unavailable preventing access to job result"
                            )

                    elif current_state == STATE_RETRIEVED_CACHED:
                        raise HTTPException(
                            status_code=410, detail="Job result is gone (previously cached, fetch failed)"
                        )

                    elif current_state in INTERMEDIATE_STATES:
                        if isinstance(fetch_err, TimeoutError):
                            raise HTTPException(
                                status_code=202, detail=f"Job is processing (state: {current_state}). Retry later"
                            )
                        else:
                            raise HTTPException(
                                status_code=503, detail="Backend service unavailable preventing fetch of job result"
                            )

                    else:
                        logger.error(f"Unexpected state '{current_state}' for job {job_id} after fetch failure")
                        raise HTTPException(
                            status_code=500, detail="Internal server error: Unexpected job state after fetch failure"
                        )

                except ValueError as ve:
                    logger.exception(f"Value error fetching job {job_id}: {ve}")
                    raise HTTPException(status_code=500, detail="Internal server error processing job data")

                except Exception as fetch_ex:
                    logger.exception(f"Unexpected error fetching job {job_id}: {fetch_ex}")
                    raise HTTPException(status_code=500, detail="Internal server error during data fetch")

            else:
                logger.error(f"Unknown job state '{current_state}' for job {job_id}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: Unknown job state '{current_state}'"
                )

        except HTTPException:
            raise

        except Exception as initial_err:
            logger.exception(f"Unexpected server error handling fetch for job {job_id}: {initial_err}")
            raise HTTPException(status_code=500, detail="Internal Server Error during job fetch")
