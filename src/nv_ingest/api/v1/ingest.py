# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

from io import BytesIO
from typing import Annotated, Dict, List, Optional
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

from nv_ingest_api.util.converters.formats import ingest_json_results_to_blob

from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask
from nv_ingest_client.primitives.tasks.infographic_extraction import InfographicExtractionTask

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()


async def _get_ingest_service() -> IngestServiceMeta:
    """
    Gather the appropriate Ingestion Service to use for the nv-ingest endpoint.
    """
    logger.debug("Creating RedisIngestService singleton for dependency injection")
    return RedisIngestService.get_instance()


INGEST_SERVICE_T = Annotated[IngestServiceMeta, Depends(_get_ingest_service)]
STATE_RETRIEVED_DESTRUCTIVE = "RETRIEVED_DESTRUCTIVE"
STATE_RETRIEVED_NON_DESTRUCTIVE = "RETRIEVED_NON_DESTRUCTIVE"
STATE_RETRIEVED_CACHED = "RETRIEVED_CACHED"
STATE_FAILED = "FAILED"
STATE_PROCESSING = "PROCESSING"
STATE_SUBMITTED = "SUBMITTED"
INTERMEDIATE_STATES = {STATE_PROCESSING, STATE_SUBMITTED}


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
        logger.exception(f"Error submitting job: {str(ex)}")
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

            current_trace_id = span.get_span_context().trace_id
            job_id = trace_id_to_uuid(current_trace_id)

            # Add trace_id to job_spec payload
            job_spec_dict = json.loads(job_spec.payload)
            if "tracing_options" not in job_spec_dict:
                job_spec_dict["tracing_options"] = {}
            job_spec_dict["tracing_options"]["trace_id"] = str(current_trace_id)
            updated_job_spec = MessageWrapper(payload=json.dumps(job_spec_dict))

            # Add another event
            span.add_event("Finished processing")

            # Submit the job to the pipeline task queue
            await ingest_service.submit_job(updated_job_spec, job_id)  # Pass job_id used for state
            await ingest_service.set_job_state(job_id, "SUBMITTED")

            response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
            return job_id

        except Exception as ex:
            logger.exception(f"Error submitting job: {str(ex)}")
            raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


# GET /fetch_job
@router.get(
    "/fetch_job/{job_id}",
    responses={
        200: {"description": "Job result successfully retrieved."},
        202: {"description": "Job is processing or result not yet available. Retry later."},
        404: {"description": "Job ID not found or associated state has expired."},
        410: {"description": "Job result existed but is now gone (expired or retrieved destructively/cached)."},
        500: {"description": "Internal server error during fetch processing."},
        503: {"description": "Job processing failed, or backend service temporarily unavailable preventing fetch."},
    },
    tags=["Ingestion"],
    summary="Fetch the result of a previously submitted job by its job_id",
    operation_id="fetch_job",
)
async def fetch_job(job_id: str, ingest_service: INGEST_SERVICE_T):
    """
    Fetches job result, checking job state *before* attempting data retrieval.

    Distinguishes non-existent jobs (404) from expired results (410).
    """
    current_state: Optional[str] = None
    try:
        # Step 1: Get Job State
        current_state = await ingest_service.get_job_state(job_id)
        logger.debug(f"Initial state check for job {job_id}: {current_state}")

        # Step 2: Handle Terminal/Invalid States Immediately
        if current_state is None:
            logger.warning(f"No state found (or state expired) for job {job_id}. Returning 404.")
            raise HTTPException(status_code=404, detail="Job ID not found or state has expired.")

        if current_state == STATE_FAILED:
            logger.error(f"Job {job_id} state is {STATE_FAILED}. Returning 503.")
            raise HTTPException(status_code=503, detail="Job processing failed.")

        if current_state == STATE_RETRIEVED_DESTRUCTIVE:
            logger.warning(f"Job {job_id} state is {STATE_RETRIEVED_DESTRUCTIVE}. Result gone. Returning 410.")
            raise HTTPException(status_code=410, detail="Job result is gone (destructive read).")

        # If state is RETRIEVED_CACHED or RETRIEVED_NON_DESTRUCTIVE or an intermediate state,
        # we proceed to attempt the fetch. RETRIEVED_CACHED implies trying cache first internally.
        if (
            current_state in INTERMEDIATE_STATES
            or current_state == STATE_RETRIEVED_NON_DESTRUCTIVE
            or current_state == STATE_RETRIEVED_CACHED
        ):

            logger.debug(f"State {current_state} allows fetch attempt for job {job_id}.")
            # Step 3: Attempt Data Fetch
            try:
                job_response = await ingest_service.fetch_job(job_id)

                # --- Fetch Success Path ---
                logger.debug(f"Successfully fetched result for job {job_id} (initial state: {current_state}).")

                # Step 3a: Determine and Set New State & Refresh TTL (Best Effort)
                try:
                    current_fetch_mode = await ingest_service.get_fetch_mode()  # Needs implementation
                    if current_fetch_mode == FetchMode.DESTRUCTIVE:
                        target_state = STATE_RETRIEVED_DESTRUCTIVE
                    elif current_fetch_mode == FetchMode.NON_DESTRUCTIVE:
                        target_state = STATE_RETRIEVED_NON_DESTRUCTIVE
                    elif current_fetch_mode == FetchMode.CACHE_BEFORE_DELETE:
                        target_state = STATE_RETRIEVED_CACHED
                    else:
                        target_state = "RETRIEVED_UNKNOWN"

                    if target_state != "RETRIEVED_UNKNOWN":
                        await ingest_service.set_job_state(job_id, target_state)  # Sets state & refreshes TTL
                        logger.debug(f"Set job state to {target_state} and refreshed state TTL for {job_id}.")

                except Exception as state_err:
                    logger.error(
                        f"Failed to set job state/refresh TTL for {job_id} after fetch: {state_err}. "
                        "Proceeding with response."
                    )

                # Step 3b: Serialize and return success response.
                try:
                    json_bytes = json.dumps(job_response).encode("utf-8")
                    return StreamingResponse(iter([json_bytes]), media_type="application/json", status_code=200)
                except TypeError as json_err:
                    logger.exception(f"Failed to serialize fetched job response for {job_id}: {json_err}")
                    raise HTTPException(
                        status_code=500, detail="Internal server error: Failed to serialize job result."
                    )

            except (TimeoutError, RedisError, ConnectionError) as fetch_err:
                # --- Fetch Failure Path ---
                fetch_err_type = type(fetch_err).__name__
                logger.warning(
                    f"Fetch attempt failed for job {job_id} ({fetch_err_type}) "
                    f"after initial state check ({current_state}). Determining final status."
                )

                # Step 3c: Evaluate based on the *original state* when fetch fails
                if current_state == STATE_RETRIEVED_NON_DESTRUCTIVE:
                    if isinstance(fetch_err, TimeoutError):
                        logger.warning(
                            f"Fetch timed out for {STATE_RETRIEVED_NON_DESTRUCTIVE} job {job_id}. "
                            f"Assuming TTL expired. Returning 410."
                        )
                        raise HTTPException(status_code=410, detail="Job result is gone (TTL expired).")
                    else:  # RedisError / ConnectionError
                        logger.error(
                            f"Backend error ({fetch_err_type}) fetching {STATE_RETRIEVED_NON_DESTRUCTIVE} "
                            f"job {job_id}. Returning 503."
                        )
                        raise HTTPException(
                            status_code=503, detail="Backend service unavailable preventing access to job result."
                        )

                elif current_state == STATE_RETRIEVED_CACHED:
                    # Fetch failed (cache miss/err + redis miss/err). Data unavailable this time.
                    logger.warning(
                        f"Fetch failed for {STATE_RETRIEVED_CACHED} job {job_id}. Result unavailable. Returning 410."
                    )
                    raise HTTPException(status_code=410, detail="Job result is gone (previously cached, fetch failed).")

                elif current_state in INTERMEDIATE_STATES:
                    if isinstance(fetch_err, TimeoutError):
                        # Job is genuinely still processing or result write delayed.
                        logger.debug(f"Fetch timed out for job {job_id} in state {current_state}. Returning 202.")
                        raise HTTPException(
                            status_code=202, detail=f"Job is processing (state: {current_state}). Retry later."
                        )
                    else:  # RedisError / ConnectionError
                        # Backend issue prevented fetch of potentially ready job.
                        logger.error(
                            f"Backend error ({fetch_err_type}) fetching job {job_id} in state {current_state}. "
                            f"Returning 503."
                        )
                        raise HTTPException(
                            status_code=503, detail="Backend service unavailable preventing fetch of job result."
                        )
                else:
                    # Should not happen if initial state checks are exhaustive, but acts as a fallback.
                    logger.error(
                        f"Unexpected state ({current_state}) encountered after fetch failure for {job_id}. "
                        f"Returning 500."
                    )
                    raise HTTPException(
                        status_code=500, detail="Internal server error: Unexpected job state after fetch failure."
                    )

            except ValueError as ve:  # Catch fetch value errors (e.g., JSON decode)
                logger.exception(f"Data error during fetch attempt for job {job_id}: {ve}")
                raise HTTPException(status_code=500, detail=f"Internal server error processing job data: {str(ve)}")
            except Exception as fetch_ex:  # Catch other unexpected fetch errors
                logger.exception(f"Unexpected error during fetch attempt for job {job_id}: {fetch_ex}")
                raise HTTPException(status_code=500, detail=f"Internal server error during data fetch: {str(fetch_ex)}")

        else:
            # State was something other than None, FAILED, RETRIEVED_*, or INTERMEDIATE_*
            # This indicates an unknown or unexpected state stored in the system.
            logger.error(f"Encountered unknown state '{current_state}' for job {job_id}. Returning 500.")
            raise HTTPException(status_code=500, detail=f"Internal server error: Unknown job state '{current_state}'.")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions raised explicitly.
        raise http_exc
    except Exception as initial_err:
        # Handle errors during the initial state check itself, or other unexpected issues.
        logger.exception(f"Unexpected error processing request for job {job_id}: {initial_err}")
        raise HTTPException(status_code=500, detail="Internal Server Error: An unexpected error occurred.")


@router.post("/convert")
async def convert_pdf(
    ingest_service: INGEST_SERVICE_T,
    files: List[UploadFile] = File(...),
    job_id: str = Form(...),
    extract_text: bool = Form(True),
    extract_images: bool = Form(True),
    extract_tables: bool = Form(True),
    extract_charts: bool = Form(False),
    extract_infographics: bool = Form(False),
) -> Dict[str, str]:
    try:

        if job_id is None:
            job_id = str(uuid.uuid4())
            logger.debug(f"JobId is None, Created JobId: {job_id}")

        submitted_jobs: List[ProcessingJob] = []
        for file in files:
            file_stream = BytesIO(file.file.read())
            doc_content = base64.b64encode(file_stream.read()).decode("utf-8")

            try:
                content_type = file.content_type.split("/")[1]
            except Exception:
                err_message = f"Unsupported content_type: {file.content_type}"
                logger.error(err_message)
                raise HTTPException(status_code=500, detail=err_message)

            job_spec = JobSpec(
                document_type=content_type,
                payload=doc_content,
                source_id=file.filename,
                source_name=file.filename,
                extended_options={
                    "tracing_options": {
                        "trace": True,
                        "ts_send": time.time_ns(),
                    }
                },
            )

            extract_task = ExtractTask(
                document_type=content_type,
                extract_text=extract_text,
                extract_images=extract_images,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_infographics=extract_infographics,
            )

            job_spec.add_task(extract_task)

            # Conditionally add tasks as needed.
            if extract_tables:
                table_data_extract = TableExtractionTask()
                job_spec.add_task(table_data_extract)

            if extract_charts:
                chart_data_extract = ChartExtractionTask()
                job_spec.add_task(chart_data_extract)

            if extract_infographics:
                infographic_data_extract = InfographicExtractionTask()
                job_spec.add_task(infographic_data_extract)

            submitted_job_id = await ingest_service.submit_job(
                MessageWrapper(payload=json.dumps(job_spec.to_dict())), job_id
            )

            processing_job = ProcessingJob(
                submitted_job_id=submitted_job_id,
                filename=file.filename,
                status=ConversionStatus.IN_PROGRESS,
            )

            submitted_jobs.append(processing_job)

        await ingest_service.set_processing_cache(job_id, submitted_jobs)

        logger.debug(f"Submitted: {len(submitted_jobs)} documents of type: '{content_type}' for processing")

        return {
            "task_id": job_id,
            "status": "processing",
            "status_url": f"/status/{job_id}",
        }

    except Exception as e:
        logger.error(f"Error starting conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_status(ingest_service: INGEST_SERVICE_T, job_id: str):
    t_start = time.time()
    try:
        processing_jobs = await ingest_service.get_processing_cache(job_id)
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    updated_cache: List[ProcessingJob] = []
    num_ready_docs = 0

    for processing_job in processing_jobs:
        logger.debug(f"submitted_job_id: {processing_job.submitted_job_id} - Status: {processing_job.status}")

        if processing_job.status == ConversionStatus.IN_PROGRESS:
            # Attempt to fetch the job from the ingest service
            try:
                job_response = await ingest_service.fetch_job(processing_job.submitted_job_id)

                job_response = json.dumps(job_response)

                # Convert JSON into pseudo markdown format
                blob_response = ingest_json_results_to_blob(job_response)

                processing_job.raw_result = job_response
                processing_job.content = blob_response
                processing_job.status = ConversionStatus.SUCCESS
                num_ready_docs = num_ready_docs + 1
                updated_cache.append(processing_job)

            except TimeoutError:
                logger.error(f"TimeoutError getting result for job_id: {processing_job.submitted_job_id}")
                updated_cache.append(processing_job)
                continue
            except RedisError:
                logger.error(f"RedisError getting result for job_id: {processing_job.submitted_job_id}")
                updated_cache.append(processing_job)
                continue
        else:
            logger.debug(f"{processing_job.submitted_job_id} has already finished successfully ....")
            num_ready_docs = num_ready_docs + 1
            updated_cache.append(processing_job)

    await ingest_service.set_processing_cache(job_id, updated_cache)

    logger.debug(f"{num_ready_docs}/{len(updated_cache)} complete")
    if num_ready_docs == len(updated_cache):
        results = []
        raw_results = []
        for result in updated_cache:
            results.append(
                {
                    "filename": result.filename,
                    "status": "success",
                    "content": result.content,
                }
            )
            raw_results.append(result.raw_result)

        return JSONResponse(
            content={"status": "completed", "result": results},
            status_code=200,
        )
    else:
        # Not yet ready ...
        logger.debug(f"/status/{job_id} endpoint execution time: {time.time() - t_start}")
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")
