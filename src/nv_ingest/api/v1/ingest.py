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

from io import BytesIO
from typing import Annotated, Dict, List
import base64
import json
import logging
import time
import traceback
import uuid

from fastapi import APIRouter, Request, Response
from fastapi import Depends
from fastapi import File, UploadFile, Form
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks.extract import ExtractTask
from opentelemetry import trace
from redis import RedisError

from nv_ingest.util.converters.formats import ingest_json_results_to_blob

from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.schemas.processing_job_schema import ConversionStatus, ProcessingJob
from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask

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


@router.post("/convert")
async def convert_pdf(
    ingest_service: INGEST_SERVICE_T,
    files: List[UploadFile] = File(...),
    job_id: str = Form(...),
    extract_text: bool = Form(True),
    extract_images: bool = Form(True),
    extract_tables: bool = Form(True),
    extract_charts: bool = Form(False),
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
            )

            job_spec.add_task(extract_task)

            # Conditionally add tasks as needed.
            if extract_tables:
                table_data_extract = TableExtractionTask()
                job_spec.add_task(table_data_extract)

            if extract_charts:
                chart_data_extract = ChartExtractionTask()
                job_spec.add_task(chart_data_extract)

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
