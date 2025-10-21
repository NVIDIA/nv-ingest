# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

import asyncio
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import logging
import os
import time
import uuid

from fastapi import APIRouter, Request, Response
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
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
from .. import traced_endpoint
from opentelemetry import trace

logger = logging.getLogger("uvicorn")

router = APIRouter()

DEFAULT_PDF_SPLIT_PAGE_COUNT = 32


def get_pdf_split_page_count(client_override: Optional[int] = None) -> int:
    """
    Resolve the page chunk size for PDF splitting with client override support.

    Priority: client_override (clamped) > env var > default (32)
    Enforces boundaries: min=1, max=128
    """
    MIN_PAGES = 1
    MAX_PAGES = 128

    # Client override takes precedence if provided
    if client_override is not None:
        clamped = max(MIN_PAGES, min(client_override, MAX_PAGES))
        if clamped != client_override:
            logger.warning(
                "Client requested split_page_count=%s; clamped to %s (min=%s, max=%s)",
                client_override,
                clamped,
                MIN_PAGES,
                MAX_PAGES,
            )
        return clamped

    # Fall back to environment variable
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

    subjob_spec = {
        key: value
        for key, value in job_spec_template.items()
        if key not in {"job_payload", "job_id", "tracing_options"}
    }

    subjob_payload_template = job_spec_template.get("job_payload", {})
    subjob_payload = {
        key: value
        for key, value in subjob_payload_template.items()
        if key not in {"content", "source_id", "source_name"}
    }

    chunk_bytes = chunk["bytes"]
    subjob_payload["content"] = [base64.b64encode(chunk_bytes).decode("utf-8")]

    page_suffix = f"page_{start_page}" if start_page == end_page else f"pages_{start_page}-{end_page}"
    subjob_payload["source_id"] = [f"{original_source_id}#{page_suffix}"]
    subjob_payload["source_name"] = [f"{original_source_name}#{page_suffix}"]

    subjob_uuid = uuid.uuid5(parent_uuid, f"chunk-{chunk_number}")
    subjob_id = str(subjob_uuid)
    subjob_spec["job_payload"] = subjob_payload
    subjob_spec["job_id"] = subjob_id

    base_tracing_options = job_spec_template.get("tracing_options") or {}
    tracing_options = dict(base_tracing_options)
    tracing_options.setdefault("trace", True)
    tracing_options["trace_id"] = str(current_trace_id)
    tracing_options["ts_send"] = int(time.time() * 1000)
    tracing_options["parent_job_id"] = parent_job_id
    tracing_options["page_num"] = start_page

    subjob_spec["tracing_options"] = tracing_options

    return subjob_id, MessageWrapper(payload=json.dumps(subjob_spec))


# ============================================================================
# Helper Functions for Fetch Job Aggregation
# ============================================================================


async def _gather_in_batches(coroutines: List, batch_size: int, return_exceptions: bool = False) -> List[Any]:
    """
    Execute coroutines in batches to respect concurrency limits.

    Parameters
    ----------
    coroutines : List
        List of coroutines to execute
    batch_size : int
        Maximum number of coroutines to execute concurrently
    return_exceptions : bool
        Whether to return exceptions as results (passed to asyncio.gather)

    Returns
    -------
    List[Any]
        Results from all coroutines in original order
    """
    results: List[Any] = []
    for offset in range(0, len(coroutines), batch_size):
        batch = coroutines[offset : offset + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=return_exceptions)
        results.extend(batch_results)
    return results


async def _update_job_state_after_fetch(job_id: str, ingest_service: INGEST_SERVICE_T) -> None:
    """
    Update job state after successful fetch based on configured fetch mode.

    Parameters
    ----------
    job_id : str
        The job identifier
    ingest_service : IngestServiceMeta
        The ingest service instance
    """
    try:
        current_fetch_mode = await ingest_service.get_fetch_mode()
        if current_fetch_mode == FetchMode.DESTRUCTIVE:
            target_state = STATE_RETRIEVED_DESTRUCTIVE
        elif current_fetch_mode == FetchMode.NON_DESTRUCTIVE:
            target_state = STATE_RETRIEVED_NON_DESTRUCTIVE
        else:
            target_state = STATE_RETRIEVED_CACHED

        await ingest_service.set_job_state(job_id, target_state)
        logger.debug(f"Updated job {job_id} state to {target_state}")
    except Exception as e:
        logger.error(f"Failed to update job state for {job_id}: {e}")


def _stream_json_response(data: Dict[str, Any]) -> StreamingResponse:
    """
    Create a StreamingResponse for JSON data.

    Parameters
    ----------
    data : Dict[str, Any]
        The data to serialize and stream

    Returns
    -------
    StreamingResponse
        FastAPI streaming response with JSON content
    """
    json_bytes = json.dumps(data).encode("utf-8")
    return StreamingResponse(iter([json_bytes]), media_type="application/json", status_code=200)


async def _check_all_subjob_states(
    ordered_descriptors: List[Dict[str, Any]], max_parallel_ops: int, ingest_service: INGEST_SERVICE_T
) -> Tuple[List[Optional[str]], List[Dict[str, object]]]:
    """
    Check the state of all subjobs in parallel batches.

    Parameters
    ----------
    ordered_descriptors : List[Dict[str, Any]]
        List of subjob descriptors with job_id and chunk_index
    max_parallel_ops : int
        Maximum number of parallel operations
    ingest_service : IngestServiceMeta
        The ingest service instance

    Returns
    -------
    Tuple[List[Optional[str]], List[Dict[str, object]]]
        Tuple of (subjob_states, failed_subjobs_list)

    Raises
    ------
    HTTPException
        If any subjob is still processing (202)
    """
    # Gather all subjob states in parallel batches
    state_coroutines = [ingest_service.get_job_state(descriptor.get("job_id")) for descriptor in ordered_descriptors]
    subjob_states = await _gather_in_batches(state_coroutines, max_parallel_ops)

    # Check for failures and pending work
    failed_subjobs: List[Dict[str, object]] = []

    for page_index, (descriptor, subjob_state) in enumerate(zip(ordered_descriptors, subjob_states), start=1):
        subjob_id = descriptor.get("job_id")

        if subjob_state == STATE_FAILED:
            logger.warning(f"Subjob {subjob_id} failed")
            failed_subjobs.append({"subjob_id": subjob_id, "chunk_index": page_index})
        elif subjob_state in INTERMEDIATE_STATES:
            raise HTTPException(status_code=202, detail="Parent job still processing. Some pages not complete.")

    return subjob_states, failed_subjobs


async def _fetch_all_subjob_results(
    ordered_descriptors: List[Dict[str, Any]],
    subjob_states: List[Optional[str]],
    failed_subjobs: List[Dict[str, object]],
    max_parallel_ops: int,
    ingest_service: INGEST_SERVICE_T,
) -> List[Optional[Dict[str, Any]]]:
    """
    Fetch results for all completed subjobs in parallel batches.

    Parameters
    ----------
    ordered_descriptors : List[Dict[str, Any]]
        List of subjob descriptors
    subjob_states : List[Optional[str]]
        States of all subjobs (from _check_all_subjob_states)
    failed_subjobs : List[Dict[str, object]]
        List to append failed fetch attempts to (modified in place)
    max_parallel_ops : int
        Maximum number of parallel operations
    ingest_service : IngestServiceMeta
        The ingest service instance

    Returns
    -------
    List[Optional[Dict[str, Any]]]
        Results for each subjob (None for failed ones)

    Raises
    ------
    HTTPException
        If any subjob is not ready yet (202)
    """
    # Initialize results array with None placeholders
    subjob_results: List[Optional[Dict[str, Any]]] = [None] * len(ordered_descriptors)

    # Build list of fetch tasks (only for non-failed subjobs)
    fetch_coroutines = []
    fetch_targets: List[Dict[str, Any]] = []

    for list_index, (page_index, descriptor, subjob_state) in enumerate(
        zip(range(1, len(ordered_descriptors) + 1), ordered_descriptors, subjob_states)
    ):
        subjob_id = descriptor.get("job_id")

        # Skip failed subjobs (already recorded in failed_subjobs)
        if subjob_state == STATE_FAILED:
            continue

        # Skip intermediate states (should have been caught earlier, but defensive)
        if subjob_state in INTERMEDIATE_STATES:
            continue

        # Queue this subjob for fetching
        fetch_coroutines.append(ingest_service.fetch_job(subjob_id))
        fetch_targets.append(
            {
                "list_index": list_index,
                "page_index": page_index,
                "subjob_id": subjob_id,
            }
        )

    # Fetch all results in parallel batches
    if fetch_coroutines:
        fetch_results = await _gather_in_batches(fetch_coroutines, max_parallel_ops, return_exceptions=True)

        # Process results and handle errors
        for target, fetch_result in zip(fetch_targets, fetch_results):
            subjob_id = target["subjob_id"]
            page_index = target["page_index"]
            list_index = target["list_index"]

            if isinstance(fetch_result, TimeoutError):
                logger.debug(f"Subjob {subjob_id} not ready yet; deferring aggregation")
                raise HTTPException(status_code=202, detail="Parent job still processing. Some pages not complete.")

            if isinstance(fetch_result, Exception):
                logger.error(f"Failed to fetch subjob {subjob_id}: {fetch_result}")
                failed_subjobs.append(
                    {
                        "subjob_id": subjob_id,
                        "chunk_index": page_index,
                        "error": str(fetch_result),
                    }
                )
                continue

            subjob_results[list_index] = fetch_result

    return subjob_results


def _extract_ray_telemetry(result: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return the trace and annotation dictionaries emitted by the sink stage."""

    if not isinstance(result, dict):
        return None, None

    trace = result.get("trace")
    annotations = result.get("annotations")

    trace_dict = trace if isinstance(trace, dict) else None
    annotations_dict = annotations if isinstance(annotations, dict) else None

    return trace_dict, annotations_dict


def _build_aggregated_response(
    parent_job_id: str,
    subjob_results: List[Optional[Dict[str, Any]]],
    failed_subjobs: List[Dict[str, object]],
    ordered_descriptors: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the aggregated response from subjob results.

    Parameters
    ----------
    parent_job_id : str
        The parent job identifier
    subjob_results : List[Optional[Dict[str, Any]]]
        Results from all subjobs (None for failed ones)
    failed_subjobs : List[Dict[str, object]]
        List of failed subjob information
    ordered_descriptors : List[Dict[str, Any]]
        Subjob descriptors in original order
    metadata : Dict[str, Any]
        Parent job metadata

    Returns
    -------
    Dict[str, Any]
        Aggregated response with combined data and metadata
    """
    any_failed = len(failed_subjobs) > 0
    subjob_ids = [desc.get("job_id") for desc in ordered_descriptors]

    aggregated_result = {
        "data": [],
        "status": "failed" if any_failed else "success",
        "description": (
            "One or more subjobs failed to complete" if any_failed else "Aggregated result composed from subjob outputs"
        ),
        "metadata": {
            "parent_job_id": parent_job_id,
            "total_pages": metadata.get("total_pages", len(subjob_ids)),
            "pages_per_chunk": metadata.get("pages_per_chunk"),
            "original_source_id": metadata.get("original_source_id"),
            "original_source_name": metadata.get("original_source_name"),
            "subjobs_failed": sum(1 for r in subjob_results if r is None),
            "failed_subjobs": failed_subjobs,
            "subjob_ids": subjob_ids,
            "chunks": [],
            "trace_segments": [],
            "annotation_segments": [],
        },
    }

    # Aggregate subjob data in page order
    for page_num, (result, descriptor) in enumerate(zip(subjob_results, ordered_descriptors), 1):
        if result is not None:
            # Add page data to aggregated result
            if "data" in result:
                aggregated_result["data"].extend(result["data"])
            chunk_entry = dict(descriptor)
            aggregated_result["metadata"]["chunks"].append(chunk_entry)

            trace_data, annotation_data = _extract_ray_telemetry(result)
            start_page = descriptor.get("start_page")
            end_page = descriptor.get("end_page")

            if trace_data:
                aggregated_result["metadata"]["trace_segments"].append(
                    {
                        "job_id": descriptor.get("job_id"),
                        "chunk_index": descriptor.get("chunk_index"),
                        "start_page": start_page,
                        "end_page": end_page,
                        "trace": trace_data,
                    }
                )

            if annotation_data:
                aggregated_result["metadata"]["annotation_segments"].append(
                    {
                        "job_id": descriptor.get("job_id"),
                        "chunk_index": descriptor.get("chunk_index"),
                        "start_page": start_page,
                        "end_page": end_page,
                        "annotations": annotation_data,
                    }
                )
        else:
            # Note failed page
            logger.warning(f"Page {page_num} failed or missing")

    return aggregated_result


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
@traced_endpoint("http-submit-job-v2")
async def submit_job_v2(
    request: Request, response: Response, job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T
):
    span = trace.get_current_span()
    try:
        span.add_event("Submitting file for processing (V2)")

        current_trace_id = span.get_span_context().trace_id
        parent_job_id = trace_id_to_uuid(current_trace_id)

        # Parse job spec
        job_spec_dict = json.loads(job_spec.payload)

        # Extract PDF configuration if provided by client
        pdf_config = job_spec_dict.get("pdf_config", {})
        client_split_page_count = pdf_config.get("split_page_count") if pdf_config else None

        # Extract document type and payload from the proper structure
        job_payload = job_spec_dict.get("job_payload", {})
        document_types = job_payload.get("document_type", [])
        payloads = job_payload.get("content", [])

        # Resolve original source metadata up front for logging / subjob naming
        source_ids = job_payload.get("source_id", ["unknown_source.pdf"])
        source_names = job_payload.get("source_name", ["unknown_source.pdf"])
        original_source_id = source_ids[0] if source_ids else "unknown_source.pdf"
        original_source_name = source_names[0] if source_names else "unknown_source.pdf"

        # Check if this is a PDF that needs splitting
        if document_types and payloads and document_types[0].lower() == "pdf":
            # Decode the payload to check page count
            pdf_content = base64.b64decode(payloads[0])
            page_count = get_pdf_page_count(pdf_content)
            pages_per_chunk = get_pdf_split_page_count(client_override=client_split_page_count)

            # Split if the document has more pages than our chunk size
            if page_count > pages_per_chunk:
                logger.warning(
                    "Splitting PDF %s into %s-page chunks (total pages: %s)",
                    original_source_name,
                    pages_per_chunk,
                    page_count,
                )

                chunks = split_pdf_to_chunks(pdf_content, pages_per_chunk)

                subjob_ids: List[str] = []
                subjob_descriptors: List[Dict[str, Any]] = []
                submission_tasks = []

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
                    subjob_descriptors.append(
                        {
                            "job_id": subjob_id,
                            "chunk_index": len(subjob_descriptors) + 1,
                            "start_page": chunk.get("start_page"),
                            "end_page": chunk.get("end_page"),
                            "page_count": chunk.get("page_count"),
                        }
                    )

                if submission_tasks:
                    await asyncio.gather(*submission_tasks)

                parent_metadata: Dict[str, Any] = {
                    "total_pages": page_count,
                    "pages_per_chunk": pages_per_chunk,
                    "original_source_id": original_source_id,
                    "original_source_name": original_source_name,
                    "document_type": document_types[0] if document_types else "pdf",
                    "subjob_order": subjob_ids,
                }

                await ingest_service.set_parent_job_mapping(
                    parent_job_id,
                    subjob_ids,
                    parent_metadata,
                    subjob_descriptors=subjob_descriptors,
                )

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
            # Not a parent job, fetch identical to V1
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
                        logger.debug(
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
            # This is a parent job - orchestrate aggregation using declarative helpers
            subjob_ids = subjob_info.get("subjob_ids", [])
            metadata = subjob_info.get("metadata", {})

            logger.debug(f"Parent job {job_id} has {len(subjob_ids)} subjobs")

            # Build ordered descriptors for subjobs
            stored_descriptors = subjob_info.get("subjob_descriptors") or []
            descriptor_lookup = {entry.get("job_id"): entry for entry in stored_descriptors if isinstance(entry, dict)}

            ordered_descriptors: List[Dict[str, Any]] = []
            for idx, subjob_id in enumerate(subjob_ids, 1):
                descriptor = descriptor_lookup.get(subjob_id, {})
                ordered_descriptors.append(
                    {
                        "job_id": subjob_id,
                        "chunk_index": descriptor.get("chunk_index", idx),
                        "start_page": descriptor.get("start_page"),
                        "end_page": descriptor.get("end_page"),
                        "page_count": descriptor.get("page_count"),
                    }
                )

            # Calculate max parallel operations (stay within Redis connection pool)
            max_parallel_ops = max(
                1, min(len(ordered_descriptors), getattr(ingest_service, "_concurrency_level", 10) // 2)
            )

            # Check all subjob states (raises 202 if any still processing)
            subjob_states, failed_subjobs = await _check_all_subjob_states(
                ordered_descriptors, max_parallel_ops, ingest_service
            )

            # Fetch all subjob results (raises 202 if any not ready)
            subjob_results = await _fetch_all_subjob_results(
                ordered_descriptors, subjob_states, failed_subjobs, max_parallel_ops, ingest_service
            )

            # Build aggregated response from all subjob results
            aggregated_result = _build_aggregated_response(
                job_id, subjob_results, failed_subjobs, ordered_descriptors, metadata
            )

            # Update parent job state after successful aggregation
            await _update_job_state_after_fetch(job_id, ingest_service)

            # Return aggregated result as streaming response
            return _stream_json_response(aggregated_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in fetch_job_v2: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during job fetch.")
