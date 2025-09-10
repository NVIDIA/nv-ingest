# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PDF Splitting Utility for V2 API

Handles splitting large PDFs into smaller chunks for improved parallel processing.
Uses pypdfium2 for robust PDF manipulation while preserving content quality.
"""

import base64
import logging
from io import BytesIO
from typing import List, Dict, Any, Tuple
import os

import pypdfium2 as pdfium
from nv_ingest_client.primitives.jobs.job_spec import JobSpec

logger = logging.getLogger("uvicorn")

# Configuration
PDF_SPLIT_THRESHOLD = int(os.getenv("PDF_SPLIT_THRESHOLD", "4"))
PDF_PAGES_PER_SUBJOB = int(os.getenv("PDF_PAGES_PER_SUBJOB", "3"))


class PDFSplitResult:
    """Result of PDF splitting operation"""

    def __init__(self, should_split: bool, total_pages: int, subjobs: List[Dict[str, Any]] = None):
        self.should_split = should_split
        self.total_pages = total_pages
        self.subjobs = subjobs or []


def get_pdf_page_count(pdf_content: bytes) -> int:
    """
    Get the number of pages in a PDF without full processing.

    Args:
        pdf_content: PDF file content as bytes

    Returns:
        Number of pages in the PDF

    Raises:
        Exception: If PDF cannot be processed
    """
    try:
        pdf_document = pdfium.PdfDocument(pdf_content)
        page_count = len(pdf_document)
        pdf_document.close()
        return page_count
    except Exception as e:
        logger.error(f"Failed to get PDF page count: {e}")
        raise


def split_pdf_into_chunks(pdf_content: bytes, pages_per_chunk: int = PDF_PAGES_PER_SUBJOB) -> List[bytes]:
    """
    Split PDF content into smaller chunks with specified pages per chunk.

    Args:
        pdf_content: Original PDF content as bytes
        pages_per_chunk: Number of pages per output chunk

    Returns:
        List of PDF chunks as bytes

    Raises:
        Exception: If PDF splitting fails
    """
    try:
        # Open source PDF
        source_pdf = pdfium.PdfDocument(pdf_content)
        total_pages = len(source_pdf)
        chunks = []

        logger.info(f"Splitting PDF with {total_pages} pages into chunks of {pages_per_chunk} pages each")

        # Create chunks
        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk, total_pages)

            # Create new PDF for this chunk
            chunk_pdf = pdfium.PdfDocument.new()

            # Import pages from source to chunk
            page_indices = list(range(start_page, end_page))
            chunk_pdf.import_pages(source_pdf, page_indices)

            # Convert to bytes
            chunk_buffer = BytesIO()
            chunk_pdf.save(chunk_buffer)
            chunks.append(chunk_buffer.getvalue())

            # Clean up chunk document
            chunk_pdf.close()

            logger.debug(f"Created chunk with pages {start_page+1}-{end_page} ({len(page_indices)} pages)")

        # Clean up source document
        source_pdf.close()

        logger.info(f"Successfully split PDF into {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to split PDF: {e}")
        raise


def create_subjob_specs(
    original_job_spec: JobSpec, pdf_chunks: List[bytes], parent_job_id: str
) -> List[Tuple[str, JobSpec]]:
    """
    Create JobSpec objects for each PDF chunk.

    Args:
        original_job_spec: Original job specification
        pdf_chunks: List of PDF chunks as bytes
        parent_job_id: Parent job identifier

    Returns:
        List of (subjob_id, JobSpec) tuples
    """
    subjobs = []

    for i, chunk_bytes in enumerate(pdf_chunks, 1):
        # Create hex-compatible subjob ID for OpenTelemetry compatibility
        # Format: Replace last 2 chars of parent UUID with 2-digit hex chunk number
        subjob_suffix = f"{i:02x}"  # Convert to 2-digit hex: 01, 02, 03, ..., 0a, 0b, etc.
        subjob_id = f"{parent_job_id[:-2]}{subjob_suffix}"

        logger.debug(f"Created hex-compatible subjob ID: {subjob_id} (chunk {i})")

        # Encode chunk as base64
        chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")

        # Create new JobSpec for this chunk
        # Note: JobSpec stores extended_options as private _extended_options, no public property
        original_extended_options = getattr(original_job_spec, "_extended_options", {})

        subjob_spec = JobSpec(
            document_type=original_job_spec.document_type,
            payload=chunk_b64,
            source_id=f"{original_job_spec.source_id}_chunk_{i}",
            source_name=f"{original_job_spec.source_name}_chunk_{i}",
            extended_options=original_extended_options.copy() if original_extended_options else {},
        )

        # Copy tasks from original job
        # Note: JobSpec stores tasks as private _tasks, no public property
        original_tasks = getattr(original_job_spec, "_tasks", [])
        for task in original_tasks:
            subjob_spec.add_task(task)

        subjobs.append((subjob_id, subjob_spec))
        logger.debug(f"Created subjob {subjob_id} for chunk {i}")

    return subjobs


def analyze_pdf_for_splitting(job_spec: JobSpec) -> PDFSplitResult:
    """
    Analyze a JobSpec to determine if PDF should be split.

    Args:
        job_spec: Job specification containing PDF content

    Returns:
        PDFSplitResult with splitting decision and metadata
    """
    try:
        # Only process PDF documents
        if job_spec.document_type.lower() != "pdf":
            logger.debug(f"Document type is {job_spec.document_type}, not PDF - skipping split")
            return PDFSplitResult(should_split=False, total_pages=1)

        logger.info(f"Analyzing PDF for splitting: document_type={job_spec.document_type}")
        if not job_spec.payload:
            logger.warning("PDF payload is empty - skipping split analysis")
            return PDFSplitResult(should_split=False, total_pages=1)

        # Decode PDF content
        try:
            pdf_content = base64.b64decode(job_spec.payload)
            logger.debug(f"Successfully decoded PDF content: {len(pdf_content)} bytes")

            # Validate PDF format
            if not pdf_content.startswith(b"%PDF"):
                logger.warning("Invalid PDF format - content doesn't start with PDF header")
                return PDFSplitResult(should_split=False, total_pages=1)

        except Exception as e:
            logger.error(f"Failed to decode PDF payload: {e}")
            return PDFSplitResult(should_split=False, total_pages=1)

        # Get page count
        total_pages = get_pdf_page_count(pdf_content)

        # Decide if splitting is needed
        should_split = total_pages > PDF_SPLIT_THRESHOLD

        if should_split:
            logger.info(f"PDF has {total_pages} pages, splitting enabled (threshold: {PDF_SPLIT_THRESHOLD})")
            return PDFSplitResult(should_split=True, total_pages=total_pages)
        else:
            logger.debug(f"PDF has {total_pages} pages, below threshold ({PDF_SPLIT_THRESHOLD})")
            return PDFSplitResult(should_split=False, total_pages=total_pages)

    except Exception as e:
        logger.error(f"Error analyzing PDF for splitting: {e}")
        # Fallback to non-split behavior
        return PDFSplitResult(should_split=False, total_pages=1)


def split_pdf_job(job_spec: JobSpec, parent_job_id: str) -> PDFSplitResult:
    """
    Split a PDF job into multiple subjobs if needed.

    Args:
        job_spec: Original job specification
        parent_job_id: Parent job identifier

    Returns:
        PDFSplitResult with subjobs if splitting occurred
    """
    try:
        # First analyze if we should split
        analysis = analyze_pdf_for_splitting(job_spec)

        if not analysis.should_split:
            return analysis

        # Decode PDF and split into chunks
        pdf_content = base64.b64decode(job_spec.payload)
        pdf_chunks = split_pdf_into_chunks(pdf_content, PDF_PAGES_PER_SUBJOB)

        # Create subjob specifications
        subjobs_list = create_subjob_specs(job_spec, pdf_chunks, parent_job_id)

        # Convert to dictionary format for Redis storage
        subjobs_dict = []
        for subjob_id, subjob_spec in subjobs_list:
            subjobs_dict.append(
                {
                    "subjob_id": subjob_id,
                    "job_spec": subjob_spec,
                    "status": "PENDING",
                    "pages": PDF_PAGES_PER_SUBJOB,  # Approximate, last chunk might have fewer
                }
            )

        result = PDFSplitResult(should_split=True, total_pages=analysis.total_pages, subjobs=subjobs_dict)

        logger.info(f"Successfully created {len(subjobs_dict)} subjobs for parent job {parent_job_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to split PDF job {parent_job_id}: {e}")
        # Return fallback to non-split processing
        return PDFSplitResult(should_split=False, total_pages=1)
