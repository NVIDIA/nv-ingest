# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

from io import BytesIO
from typing import Annotated, Dict, List, Optional
import base64
import copy
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from redis import RedisError

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.util.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.framework.util.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.primitives.tasks.extract import ExtractTask

# For PDF splitting
import pypdfium2 as pdfium

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()

# Reuse V1 state constants and helper functions
from ..v1.ingest import (
    _get_ingest_service,
    trace_id_to_uuid,
    INGEST_SERVICE_T,
    STATE_RETRIEVED_DESTRUCTIVE,
    STATE_RETRIEVED_NON_DESTRUCTIVE,
    STATE_RETRIEVED_CACHED,
    STATE_FAILED,
    STATE_PROCESSING,
    STATE_SUBMITTED,
    INTERMEDIATE_STATES,
)


def split_pdf_to_pages(pdf_content: bytes) -> List[bytes]:
    """
    Split a PDF into individual pages using pypdfium2.
    
    Parameters
    ----------
    pdf_content : bytes
        The PDF file content as bytes
        
    Returns
    -------
    List[bytes]
        List of single-page PDFs as bytes
    """
    page_pdfs = []
    
    # Load the PDF
    pdf = pdfium.PdfDocument(pdf_content)
    page_count = len(pdf)
    
    for page_index in range(page_count):
        # Create a new document with just this page
        new_pdf = pdfium.PdfDocument.new()
        new_pdf.import_pages(pdf, [page_index])
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        new_pdf.save(buffer)
        buffer.seek(0)
        page_pdfs.append(buffer.read())
        
        # Clean up the new PDF
        new_pdf.close()
    
    # Clean up original PDF
    pdf.close()
    
    return page_pdfs


def get_pdf_page_count(pdf_content: bytes) -> int:
    """Get the number of pages in a PDF using pypdfium2."""
    try:
        pdf = pdfium.PdfDocument(pdf_content)
        page_count = len(pdf)
        pdf.close()
        return page_count
    except Exception as e:
        logger.warning(f"Failed to get PDF page count: {e}")
        return 1  # Assume single page on error


# POST /v2/submit_job
@router.post(
    "/submit_job",
    responses={
        200: {"description": "Jobs were successfully submitted"},
        500: {"description": "Error encountered while submitting jobs."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="submit jobs to the core nv ingestion service for processing with PDF splitting",
    operation_id="submit_job_v2",
)
async def submit_job_v2(request: Request, response: Response, job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T):
    with tracer.start_as_current_span("http-submit-job-v2") as span:
        try:
            # Add custom attributes to the span
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.add_event("Submitting file for processing (V2)")

            current_trace_id = span.get_span_context().trace_id
            parent_job_id = trace_id_to_uuid(current_trace_id)

            # Parse job spec
            job_spec_dict = json.loads(job_spec.payload)
            
            # Extract document type and payload from the proper structure
            job_payload = job_spec_dict.get("job_payload", {})
            document_types = job_payload.get("document_type", [])
            payloads = job_payload.get("content", [])
            
            # Check if this is a PDF that needs splitting
            if document_types and payloads and document_types[0].lower() == "pdf":
                # Decode the payload to check page count
                pdf_content = base64.b64decode(payloads[0])
                page_count = get_pdf_page_count(pdf_content)
                
                # Split if more than 3 pages
                if page_count > 3:
                    logger.info(f"Splitting PDF with {page_count} pages into subjobs")
                    
                    # Split the PDF
                    page_pdfs = split_pdf_to_pages(pdf_content)
                    
                    # Create subjobs for each page
                    subjob_ids = []
                    source_ids = job_payload.get("source_id", ["document.pdf"])
                    source_names = job_payload.get("source_name", ["document.pdf"])
                    original_source_id = source_ids[0] if source_ids else "document.pdf"
                    original_source_name = source_names[0] if source_names else "document.pdf"
                    
                    try:
                        parent_uuid = uuid.UUID(parent_job_id)
                    except ValueError:
                        logger.warning(
                            "Parent job id %s is not a valid UUID; generating fallback namespace for subjobs",
                            parent_job_id,
                        )
                        parent_uuid = uuid.uuid4()

                    subjob_order = []

                    for page_num, page_pdf in enumerate(page_pdfs, 1):
                        # Create subjob spec by deep copying and modifying
                        subjob_spec = copy.deepcopy(job_spec_dict)
                        
                        # Update the payload in the proper location
                        subjob_spec["job_payload"]["content"] = [base64.b64encode(page_pdf).decode("utf-8")]
                        subjob_spec["job_payload"]["source_id"] = [f"{original_source_id}#page_{page_num}"]
                        subjob_spec["job_payload"]["source_name"] = [f"{original_source_name}#page_{page_num}"]
                        
                        # Generate telemetry-safe subjob ID derived from the parent UUID
                        subjob_uuid = uuid.uuid5(parent_uuid, f"page-{page_num}")
                        subjob_id = str(subjob_uuid)
                        subjob_spec["job_id"] = subjob_id
                        subjob_order.append(subjob_id)
                        
                        # Add tracing info with V2 fields
                        if "tracing_options" not in subjob_spec:
                            subjob_spec["tracing_options"] = {"trace": True}
                        subjob_spec["tracing_options"]["trace_id"] = str(current_trace_id)
                        subjob_spec["tracing_options"]["ts_send"] = int(time.time() * 1000)  # ms
                        subjob_spec["tracing_options"]["parent_job_id"] = parent_job_id
                        subjob_spec["tracing_options"]["page_num"] = page_num
                        subjob_spec["tracing_options"]["total_pages"] = page_count
                        
                        # Submit subjob
                        subjob_wrapper = MessageWrapper(payload=json.dumps(subjob_spec))
                        await ingest_service.submit_job(subjob_wrapper, subjob_id)
                        subjob_ids.append(subjob_id)
                    
                    # Store parent-child mapping in Redis
                    parent_metadata = {
                        "total_pages": page_count,
                        "original_source_id": original_source_id,
                        "original_source_name": original_source_name,
                        "document_type": document_types[0] if document_types else "pdf",
                    }
                    if subjob_order:
                        parent_metadata["subjob_order"] = subjob_order

                    await ingest_service.set_parent_job_mapping(parent_job_id, subjob_ids, parent_metadata)
                    
                    # Set parent job state
                    await ingest_service.set_job_state(parent_job_id, STATE_SUBMITTED)
                    
                    span.add_event(f"Split into {len(subjob_ids)} subjobs")
                    response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
                    return parent_job_id
                    
            # For non-PDFs or PDFs with <= 3 pages, submit as normal
            if "tracing_options" not in job_spec_dict:
                job_spec_dict["tracing_options"] = {"trace": True}
            job_spec_dict["tracing_options"]["trace_id"] = str(current_trace_id)
            updated_job_spec = MessageWrapper(payload=json.dumps(job_spec_dict))
            
            span.add_event("Submitting as single job (no split needed)")
            
            # Submit the job to the pipeline task queue
            await ingest_service.submit_job(updated_job_spec, parent_job_id)
            await ingest_service.set_job_state(parent_job_id, STATE_SUBMITTED)
            
            response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
            return parent_job_id

        except Exception as ex:
            logger.exception(f"Error submitting job: {str(ex)}")
            raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


# GET /v2/fetch_job
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
    summary="Fetch the result of a previously submitted job by its job_id (V2 with aggregation)",
    operation_id="fetch_job_v2",
)
async def fetch_job_v2(job_id: str, ingest_service: INGEST_SERVICE_T):
    """
    V2 fetch that handles parent job aggregation.
    """
    try:
        # Check if this is a parent job with subjobs
        subjob_info = await ingest_service.get_parent_job_info(job_id)
        
        if subjob_info is None:
            # Not a parent job, fetch normally like V1
            current_state = await ingest_service.get_job_state(job_id)
            logger.debug(f"Initial state check for job {job_id}: {current_state}")

            if current_state is None:
                logger.warning(f"Job {job_id} not found or expired. Returning 404.")
                raise HTTPException(status_code=404, detail="Job ID not found or state has expired.")

            if current_state == STATE_FAILED:
                logger.error(f"Job {job_id} failed. Returning 503.")
                raise HTTPException(status_code=503, detail="Job processing failed.")

            if current_state == STATE_RETRIEVED_DESTRUCTIVE:
                logger.warning(f"Job {job_id} was destructively retrieved. Returning 410.")
                raise HTTPException(status_code=410, detail="Job result is gone (destructive read).")

            if current_state in INTERMEDIATE_STATES or current_state in {
                STATE_RETRIEVED_NON_DESTRUCTIVE,
                STATE_RETRIEVED_CACHED,
            }:
                logger.debug(f"Attempting fetch for job {job_id} in state {current_state}.")

                try:
                    job_response = await ingest_service.fetch_job(job_id)
                    logger.debug(f"Fetched result for job {job_id}.")

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
                            logger.debug(f"Updated job {job_id} state to {target_state}.")
                    except Exception as state_err:
                        logger.error(f"Failed to set job state for {job_id} after fetch: {state_err}")

                    try:
                        json_bytes = json.dumps(job_response).encode("utf-8")
                        return StreamingResponse(iter([json_bytes]), media_type="application/json", status_code=200)
                    except TypeError as json_err:
                        logger.exception(f"Serialization error for job {job_id}: {json_err}")
                        raise HTTPException(status_code=500, detail="Internal server error: Failed to serialize result.")

                except (TimeoutError, RedisError, ConnectionError) as fetch_err:
                    # Handle timeout/error cases same as V1
                    fetch_err_type = type(fetch_err).__name__

                    if isinstance(fetch_err, TimeoutError):
                        logger.info(
                            f"Job {job_id} still processing (state: {current_state}), fetch attempt timed out cleanly."
                        )
                    else:
                        logger.warning(
                            f"Backend error ({fetch_err_type}) during fetch attempt for job {job_id} "
                            f"(state: {current_state}): {fetch_err}"
                        )

                    if current_state == STATE_RETRIEVED_NON_DESTRUCTIVE:
                        if isinstance(fetch_err, TimeoutError):
                            raise HTTPException(status_code=410, detail="Job result is gone (TTL expired).")
                        else:
                            raise HTTPException(
                                status_code=503, detail="Backend service unavailable preventing access to job result."
                            )
                    elif current_state == STATE_RETRIEVED_CACHED:
                        raise HTTPException(status_code=410, detail="Job result is gone (previously cached, fetch failed).")
                    elif current_state in INTERMEDIATE_STATES:
                        if isinstance(fetch_err, TimeoutError):
                            raise HTTPException(
                                status_code=202, detail=f"Job is processing (state: {current_state}). Retry later."
                            )
                        else:
                            raise HTTPException(
                                status_code=503, detail="Backend service unavailable preventing fetch of job result."
                            )
                    else:
                        logger.error(f"Unexpected state '{current_state}' for job {job_id} after fetch failure.")
                        raise HTTPException(
                            status_code=500, detail="Internal server error: Unexpected job state after fetch failure."
                        )
            else:
                logger.error(f"Unknown job state '{current_state}' for job {job_id}.")
                raise HTTPException(status_code=500, detail=f"Internal server error: Unknown job state '{current_state}'.")
                
        else:
            # This is a parent job - need to aggregate subjobs
            subjob_ids = subjob_info["subjob_ids"]
            metadata = subjob_info["metadata"]
            
            logger.info(f"Parent job {job_id} has {len(subjob_ids)} subjobs")
            
            # Check all subjob states
            all_complete = True
            any_failed = False
            subjob_results = []
            failed_subjobs: List[Dict[str, object]] = []
            
            for page_index, subjob_id in enumerate(subjob_ids, 1):
                subjob_state = await ingest_service.get_job_state(subjob_id)
                
                if subjob_state == STATE_FAILED:
                    any_failed = True
                    logger.warning(f"Subjob {subjob_id} failed")
                    subjob_results.append(None)  # Placeholder for failed page
                    failed_subjobs.append({"subjob_id": subjob_id, "page_num": page_index})
                elif subjob_state in INTERMEDIATE_STATES:
                    all_complete = False
                    break
                else:
                    # Try to fetch the subjob result
                    try:
                        subjob_response = await ingest_service.fetch_job(subjob_id)
                        subjob_results.append(subjob_response)
                    except TimeoutError:
                        logger.debug(f"Subjob {subjob_id} not ready yet; deferring aggregation")
                        all_complete = False
                        break
                    except Exception as e:
                        logger.error(f"Failed to fetch subjob {subjob_id}: {e}")
                        any_failed = True
                        subjob_results.append(None)
                        failed_subjobs.append({"subjob_id": subjob_id, "page_num": page_index, "error": str(e)})
            
            if not all_complete:
                # Some subjobs still processing
                raise HTTPException(status_code=202, detail="Parent job still processing. Some pages not complete.")
            
            # All subjobs complete - aggregate results
            aggregated_result = {
                "data": [],
                "status": "failed" if any_failed else "success",
                "description": (
                    "One or more subjobs failed to complete"
                    if any_failed
                    else "Aggregated result composed from subjob outputs"
                ),
                "metadata": {
                    "parent_job_id": job_id,
                    "total_pages": metadata.get("total_pages", len(subjob_ids)),
                    "original_source_id": metadata.get("original_source_id"),
                    "original_source_name": metadata.get("original_source_name"),
                    "subjobs_failed": sum(1 for r in subjob_results if r is None),
                    "failed_subjobs": failed_subjobs,
                    "subjob_ids": subjob_ids,
                },
            }
            
            # Aggregate subjob data in page order
            for page_num, result in enumerate(subjob_results, 1):
                if result is not None:
                    # Add page data to aggregated result
                    if "data" in result:
                        aggregated_result["data"].extend(result["data"])
                else:
                    # Note failed page
                    logger.warning(f"Page {page_num} failed or missing")
            
            # Update parent state
            try:
                current_fetch_mode = await ingest_service.get_fetch_mode()
                if current_fetch_mode == FetchMode.DESTRUCTIVE:
                    target_state = STATE_RETRIEVED_DESTRUCTIVE
                elif current_fetch_mode == FetchMode.NON_DESTRUCTIVE:
                    target_state = STATE_RETRIEVED_NON_DESTRUCTIVE
                else:
                    target_state = STATE_RETRIEVED_CACHED
                    
                await ingest_service.set_job_state(job_id, target_state)
            except Exception as e:
                logger.error(f"Failed to update parent job state: {e}")
            
            # Return aggregated result
            json_bytes = json.dumps(aggregated_result).encode("utf-8")
            return StreamingResponse(iter([json_bytes]), media_type="application/json", status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in fetch_job_v2: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during job fetch.")
