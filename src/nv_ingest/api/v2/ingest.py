# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

import asyncio
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import base64
import copy
import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from redis import RedisError

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest_api.util.service_clients.client_base import FetchMode

# For PDF splitting
import pypdfium2 as pdfium

# Reuse V1 state constants and helper functions
from ..v1.ingest import (
    trace_id_to_uuid,
    INGEST_SERVICE_T,
    STATE_RETRIEVED_DESTRUCTIVE,
    STATE_RETRIEVED_NON_DESTRUCTIVE,
    STATE_RETRIEVED_CACHED,
    STATE_FAILED,
    STATE_SUBMITTED,
    INTERMEDIATE_STATES,
)

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()

DEFAULT_PDF_SPLIT_PAGE_COUNT = 64


def get_pdf_split_page_count() -> int:
    """Resolve the configured page chunk size for PDF splitting."""

    raw_value = os.environ.get("PDF_SPLIT_PAGE_COUNT")
    if raw_value is None:
        return DEFAULT_PDF_SPLIT_PAGE_COUNT

    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid PDF_SPLIT_PAGE_COUNT '%s'; falling back to default %s", raw_value, DEFAULT_PDF_SPLIT_PAGE_COUNT
        )
        return DEFAULT_PDF_SPLIT_PAGE_COUNT

    if parsed <= 0:
        logger.warning("PDF_SPLIT_PAGE_COUNT must be >= 1; received %s. Using 1.", parsed)
        return 1

    return parsed


def split_pdf_to_chunks(pdf_content: bytes, pages_per_chunk: int) -> List[Dict[str, Any]]:
    """
    Split a PDF into multi-page chunks using pypdfium2.

    Returns a list of dictionaries containing the chunk bytes and page range metadata.
    Note: this currently buffers each chunk in-memory; consider streaming in future upgrades.
    """

    chunks: List[Dict[str, Any]] = []

    if pages_per_chunk <= 0:
        pages_per_chunk = 1

    pdf = pdfium.PdfDocument(pdf_content)
    total_pages = len(pdf)

    try:
        for chunk_index, start_zero in enumerate(range(0, total_pages, pages_per_chunk)):
            end_zero = min(start_zero + pages_per_chunk, total_pages)
            page_indices = list(range(start_zero, end_zero))

            new_pdf = pdfium.PdfDocument.new()
            try:
                new_pdf.import_pages(pdf, page_indices)

                buffer = BytesIO()
                try:
                    new_pdf.save(buffer)
                    chunk_bytes = buffer.getvalue()
                finally:
                    buffer.close()
            finally:
                new_pdf.close()

            start_page = start_zero + 1
            end_page = end_zero
            chunk_info: Dict[str, Any] = {
                "bytes": chunk_bytes,
                "chunk_index": chunk_index,
                "start_page": start_page,
                "end_page": end_page,
                "page_count": end_page - start_page + 1,
            }
            chunks.append(chunk_info)

    finally:
        pdf.close()

    return chunks


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


def _prepare_chunk_submission(
    job_spec_template: Dict[str, Any],
    chunk: Dict[str, Any],
    *,
    parent_uuid: uuid.UUID,
    parent_job_id: str,
    current_trace_id: int,
    original_source_id: str,
    original_source_name: str,
) -> Tuple[str, MessageWrapper]:
    """Create a subjob MessageWrapper for a PDF chunk and return its identifier."""

    chunk_number = chunk["chunk_index"] + 1
    start_page = chunk["start_page"]
    end_page = chunk["end_page"]

    subjob_spec = copy.deepcopy(job_spec_template)
    subjob_payload = subjob_spec.setdefault("job_payload", {})

    chunk_bytes = chunk["bytes"]
    subjob_payload["content"] = [base64.b64encode(chunk_bytes).decode("utf-8")]

    page_suffix = f"page_{start_page}" if start_page == end_page else f"pages_{start_page}-{end_page}"
    subjob_payload["source_id"] = [f"{original_source_id}#{page_suffix}"]
    subjob_payload["source_name"] = [f"{original_source_name}#{page_suffix}"]

    subjob_uuid = uuid.uuid5(parent_uuid, f"chunk-{chunk_number}")
    subjob_id = str(subjob_uuid)
    subjob_spec["job_id"] = subjob_id

    tracing_options = subjob_spec.setdefault("tracing_options", {"trace": True})
    tracing_options["trace_id"] = str(current_trace_id)
    tracing_options["ts_send"] = int(time.time() * 1000)
    tracing_options["parent_job_id"] = parent_job_id
    tracing_options["page_num"] = start_page

    return subjob_id, MessageWrapper(payload=json.dumps(subjob_spec))


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
async def submit_job_v2(
    request: Request, response: Response, job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T
):
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
                pages_per_chunk = get_pdf_split_page_count()

                # Split if the document has more pages than our chunk size
                if page_count > pages_per_chunk:
                    logger.info(
                        "Splitting PDF with %s pages into chunks of %s",
                        page_count,
                        pages_per_chunk,
                    )

                    chunks = split_pdf_to_chunks(pdf_content, pages_per_chunk)

                    subjob_ids: List[str] = []
                    submission_tasks = []
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

                    for chunk in chunks:
                        subjob_id, subjob_wrapper = _prepare_chunk_submission(
                            job_spec_dict,
                            chunk,
                            parent_uuid=parent_uuid,
                            parent_job_id=parent_job_id,
                            current_trace_id=current_trace_id,
                            original_source_id=original_source_id,
                            original_source_name=original_source_name,
                        )
                        submission_tasks.append(ingest_service.submit_job(subjob_wrapper, subjob_id))
                        subjob_ids.append(subjob_id)

                    if submission_tasks:
                        await asyncio.gather(*submission_tasks)

                    parent_metadata: Dict[str, Any] = {
                        "total_pages": page_count,
                        "original_source_id": original_source_id,
                        "original_source_name": original_source_name,
                        "document_type": document_types[0] if document_types else "pdf",
                        "subjob_order": subjob_ids,
                    }

                    await ingest_service.set_parent_job_mapping(parent_job_id, subjob_ids, parent_metadata)

                    await ingest_service.set_job_state(parent_job_id, STATE_SUBMITTED)

                    span.add_event(f"Split into {len(subjob_ids)} subjobs")
                    response.headers["x-trace-id"] = trace.format_trace_id(current_trace_id)
                    return parent_job_id

            # For non-PDFs or cases where splitting is not required, submit as normal
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
                        raise HTTPException(
                            status_code=500, detail="Internal server error: Failed to serialize result."
                        )

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
                        raise HTTPException(
                            status_code=410, detail="Job result is gone (previously cached, fetch failed)."
                        )
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
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: Unknown job state '{current_state}'."
                )

        else:
            # This is a parent job - need to aggregate subjobs
            subjob_ids = subjob_info.get("subjob_ids", [])
            metadata = subjob_info.get("metadata", {})

            logger.info(f"Parent job {job_id} has {len(subjob_ids)} subjobs")

            total_pages = metadata.get("total_pages")
            if isinstance(total_pages, str):
                try:
                    total_pages = int(total_pages)
                except ValueError:
                    logger.warning("Invalid total_pages '%s' for parent %s; recomputing", total_pages, job_id)
                    total_pages = None

            ordered_descriptors: List[Dict[str, Any]] = []
            for idx, subjob_id in enumerate(subjob_ids, 1):
                ordered_descriptors.append({"job_id": subjob_id, "chunk_index": idx})

            # Limit concurrent Redis calls to stay within connection pool bounds
            max_parallel_ops = max(
                1, min(len(ordered_descriptors), getattr(ingest_service, "_concurrency_level", 10) // 2)
            )

            # Check all subjob states using bounded concurrency
            any_failed = False
            failed_subjobs: List[Dict[str, object]] = []
            subjob_results: List[Optional[Dict[str, Any]]] = [None] * len(ordered_descriptors)

            subjob_states: List[Optional[str]] = []
            for offset in range(0, len(ordered_descriptors), max_parallel_ops):
                state_batch = ordered_descriptors[offset : offset + max_parallel_ops]
                batch_tasks = [ingest_service.get_job_state(descriptor.get("job_id")) for descriptor in state_batch]
                subjob_states.extend(await asyncio.gather(*batch_tasks))

            fetch_coroutines = []
            fetch_targets: List[Dict[str, Any]] = []

            for list_index, (page_index, descriptor, subjob_state) in enumerate(
                zip(range(1, len(ordered_descriptors) + 1), ordered_descriptors, subjob_states)
            ):
                subjob_id = descriptor.get("job_id")

                if subjob_state == STATE_FAILED:
                    any_failed = True
                    logger.warning(f"Subjob {subjob_id} failed")
                    failed_subjobs.append({"subjob_id": subjob_id, "chunk_index": page_index})
                    continue

                if subjob_state in INTERMEDIATE_STATES:
                    raise HTTPException(status_code=202, detail="Parent job still processing. Some pages not complete.")

                fetch_coroutines.append(ingest_service.fetch_job(subjob_id))
                fetch_targets.append(
                    {
                        "list_index": list_index,
                        "page_index": page_index,
                        "descriptor": descriptor,
                        "subjob_id": subjob_id,
                    }
                )

            if fetch_coroutines:
                fetch_results: List[Any] = []
                for offset in range(0, len(fetch_coroutines), max_parallel_ops):
                    fetch_batch = fetch_coroutines[offset : offset + max_parallel_ops]
                    fetch_results.extend(await asyncio.gather(*fetch_batch, return_exceptions=True))

                for target, fetch_result in zip(fetch_targets, fetch_results):
                    subjob_id = target["subjob_id"]
                    page_index = target["page_index"]
                    list_index = target["list_index"]

                    if isinstance(fetch_result, TimeoutError):
                        logger.debug(f"Subjob {subjob_id} not ready yet; deferring aggregation")
                        raise HTTPException(
                            status_code=202, detail="Parent job still processing. Some pages not complete."
                        )

                    if isinstance(fetch_result, Exception):
                        logger.error(f"Failed to fetch subjob {subjob_id}: {fetch_result}")
                        any_failed = True
                        failed_entry = {
                            "subjob_id": subjob_id,
                            "chunk_index": page_index,
                            "error": str(fetch_result),
                        }
                        failed_subjobs.append(failed_entry)
                        continue

                    subjob_results[list_index] = fetch_result

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
                    "pages_per_chunk": metadata.get("pages_per_chunk"),
                    "original_source_id": metadata.get("original_source_id"),
                    "original_source_name": metadata.get("original_source_name"),
                    "subjobs_failed": sum(1 for r in subjob_results if r is None),
                    "failed_subjobs": failed_subjobs,
                    "subjob_ids": subjob_ids,
                    "chunks": [],
                },
            }

            # Aggregate subjob data in page order
            for page_num, (result, descriptor) in enumerate(zip(subjob_results, ordered_descriptors), 1):
                if result is not None:
                    # Add page data to aggregated result
                    if "data" in result:
                        aggregated_result["data"].extend(result["data"])
                    chunk_entry = dict(descriptor)
                    aggregated_result["metadata"].setdefault("chunks", []).append(chunk_entry)
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
