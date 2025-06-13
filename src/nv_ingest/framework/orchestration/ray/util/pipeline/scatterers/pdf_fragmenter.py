# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional, Callable
import time
import logging

import pandas as pd
import pypdfium2 as pdfium
from copy import deepcopy

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


def _validate_single_row(payload: pd.DataFrame) -> pd.Series:
    """Validate input DataFrame has single row."""
    if len(payload) != 1:
        raise ValueError(f"Expected single row DataFrame, got {len(payload)} rows")
    return payload.iloc[0]


def _extract_pdf_content(row: pd.Series) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extract PDF content and metadata from DataFrame row."""
    if row["document_type"] != "pdf":
        return None, None

    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]

    if "content" not in metadata:
        raise ValueError("No 'content' field found in metadata")

    return metadata["content"], metadata


def _extract_pdf_metadata(pdf: pdfium.PdfDocument) -> Dict[str, str]:
    """Extract internal PDF metadata fields."""
    pdf_metadata = {}
    metadata_keys = ["Title", "Author", "Subject", "Keywords", "Creator", "Producer", "CreationDate", "ModDate"]

    for key in metadata_keys:
        try:
            value = pdf.get_metadata_value(key)
            if value:
                pdf_metadata[key] = value
        except Exception:
            pass  # Skip if metadata field not available

    return pdf_metadata


def _create_pdf_fragment(source_pdf: pdfium.PdfDocument, start_page: int, end_page: int) -> bytes:
    """Create a PDF fragment from page range."""
    new_pdf = pdfium.PdfDocument.new()

    # Copy pages
    new_pdf.import_pages(source_pdf, list(range(start_page, end_page)))

    # Save to bytes
    buffer = BytesIO()
    new_pdf.save(buffer)
    fragment_bytes = buffer.getvalue()
    new_pdf.close()

    return fragment_bytes


def _build_fragment_info(
    fragment_idx: int,
    total_fragments: int,
    start_page: int,
    end_page: int,
    total_pages: int,
    document_type: str,
    pdf_metadata: Dict[str, str],
) -> Dict[str, Any]:
    """Build fragment info dictionary."""
    return {
        "fragment_index": fragment_idx,
        "total_fragments": total_fragments,
        "start_page": start_page,
        "end_page": end_page,
        "total_pages": total_pages,
        "pages_in_fragment": end_page - start_page,
        "source_document_type": document_type,
        "is_fragment": True,
        "fragment_id": f"{fragment_idx+1}_of_{total_fragments}",
        "original_pdf_metadata": pdf_metadata,
    }


def _build_fragment_metadata(
    original_metadata: Dict[str, Any], fragment_base64: str, fragment_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Build complete metadata for fragment."""
    new_metadata = deepcopy(original_metadata)
    new_metadata["content"] = fragment_base64
    new_metadata["fragment_info"] = fragment_info

    # Preserve original fragment info if it exists
    if "fragment_info" in original_metadata and "original_fragment_info" not in new_metadata:
        new_metadata["original_fragment_info"] = original_metadata["fragment_info"]

    return new_metadata


def _create_fragment_message(
    original_message: IngestControlMessage,
    payload: pd.DataFrame,
    new_metadata: Dict[str, Any],
    fragment_idx: int,
    total_fragments: int,
    document_type: str,
) -> IngestControlMessage:
    """Create new IngestControlMessage for fragment."""
    new_df = payload.copy(deep=True)
    new_df.at[0, "metadata"] = (
        json.dumps(new_metadata) if isinstance(payload.iloc[0]["metadata"], str) else new_metadata
    )

    new_message = deepcopy(original_message)
    new_message.payload(new_df)

    # Update message-level metadata if available
    if hasattr(new_message, "metadata"):
        msg_meta = new_message.metadata() or {}
        msg_meta.update(
            {"fragment_index": fragment_idx, "total_fragments": total_fragments, "source_document_type": document_type}
        )
        new_message.metadata(msg_meta)

    return new_message


def fragment_pdf(message: IngestControlMessage, pages_per_fragment: int = 10) -> List[IngestControlMessage]:
    """
    Fragment a PDF document into smaller PDFs based on page count.
    Preserves DataFrame metadata and extracts PDF internal metadata.

    Note: pypdfium2 does not support writing PDF metadata, so internal PDF
    metadata is only preserved in the fragment_info for reference.
    """
    # Extract and validate input
    payload = message.payload()
    row = _validate_single_row(payload)

    # Early return if not PDF
    pdf_base64, metadata = _extract_pdf_content(row)
    if pdf_base64 is None:
        return [message]

    # Load PDF
    pdf_bytes = base64.b64decode(pdf_base64)
    pdf = pdfium.PdfDocument(pdf_bytes)
    total_pages = len(pdf)

    # Check if fragmentation needed
    num_fragments = (total_pages + pages_per_fragment - 1) // pages_per_fragment
    if num_fragments == 1:
        pdf.close()
        return [message]

    # Extract PDF metadata once (for preservation in fragment_info)
    pdf_metadata = _extract_pdf_metadata(pdf)
    document_type = row["document_type"]

    # Create fragments
    output_messages = []

    for fragment_idx in range(num_fragments):
        # Calculate page range
        start_page = fragment_idx * pages_per_fragment
        end_page = min(start_page + pages_per_fragment, total_pages)

        # Create fragment PDF
        fragment_bytes = _create_pdf_fragment(pdf, start_page, end_page)

        # Build metadata
        fragment_base64 = base64.b64encode(fragment_bytes).decode("utf-8")
        fragment_info = _build_fragment_info(
            fragment_idx, num_fragments, start_page, end_page, total_pages, document_type, pdf_metadata
        )
        new_metadata = _build_fragment_metadata(metadata, fragment_base64, fragment_info)

        # Create message
        new_message = _create_fragment_message(
            message, payload, new_metadata, fragment_idx, num_fragments, document_type
        )

        output_messages.append(new_message)

    pdf.close()
    return output_messages


def create_pdf_fragmenter(
    pages_per_fragment: int = 10, actor_logger: Optional[logging.Logger] = None
) -> Callable[[IngestControlMessage], List[IngestControlMessage]]:
    """
    Factory function to create a PDF fragmenter with specific configuration using pypdfium2.

    Parameters
    ----------
    pages_per_fragment : int
        Number of pages per fragment
    actor_logger : Optional[logging.Logger]
        The logger instance from the calling Ray actor.

    Returns
    -------
    Callable[[IngestControlMessage], List[IngestControlMessage]]
        A configured PDF fragmenter function that will use the provided logger.
    """
    # Use a default logger if none is provided, though this won't be visible in Ray actor stdout by default.
    # The primary expectation is that PDFScatterStage will pass its self._logger.
    effective_logger = actor_logger if actor_logger else logging.getLogger(__name__ + ".create_pdf_fragmenter_default")
    if not effective_logger.hasHandlers() and not actor_logger:  # Only setup default if no actor_logger and no handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        effective_logger.addHandler(handler)
        effective_logger.setLevel(logging.INFO)

    # Define the configured fragmenter function within this scope to capture pages_per_fragment
    def _fragment_pdf_configured_internal(message: IngestControlMessage) -> List[IngestControlMessage]:
        # This is where the actual fragmentation logic with timing is.
        # It now uses 'effective_logger' which is captured from the outer scope.
        overall_start_time = time.perf_counter()
        effective_logger.info("_fragment_pdf_configured_internal: Starting PDF fragmentation for message.")

        initial_checks_start_time = time.perf_counter()
        payload = message.payload()

        if len(payload) != 1:
            effective_logger.error(f"Expected single row DataFrame, got {len(payload)} rows. Aborting fragmentation.")
            raise ValueError(f"Expected single row DataFrame, got {len(payload)} rows")

        row = payload.iloc[0]

        if row["document_type"] != "pdf":
            effective_logger.info(f"Document type is not PDF ('{row['document_type']}'). Skipping fragmentation.")
            return [message]
        initial_checks_end_time = time.perf_counter()
        effective_logger.info(f"Initial checks took: {initial_checks_end_time - initial_checks_start_time:.4f} seconds")

        pdf_load_start_time = time.perf_counter()
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        pdf_base64 = metadata["content"]
        pdf_bytes = base64.b64decode(pdf_base64)

        pdf = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(pdf)
        pdf_load_end_time = time.perf_counter()
        effective_logger.info(
            f"PDF loading (decode & pypdfium2) took: {pdf_load_end_time - pdf_load_start_time:.4f} seconds. "
            f"Total pages: {total_pages}"
        )

        calc_fragments_start_time = time.perf_counter()
        num_fragments = (total_pages + pages_per_fragment - 1) // pages_per_fragment
        calc_fragments_end_time = time.perf_counter()
        effective_logger.info(
            f"Fragment calculation took: {calc_fragments_end_time - calc_fragments_start_time:.4f} seconds. "
            f"Num fragments: {num_fragments}"
        )

        if num_fragments == 1:
            pdf.close()
            effective_logger.info("Single fragment, no actual fragmentation needed. Returning original message.")
            overall_end_time_no_frag = time.perf_counter()
            effective_logger.info(
                "_fragment_pdf_configured_internal: Total time (no fragmentation): "
                f"{overall_end_time_no_frag - overall_start_time:.4f} seconds"
            )
            return [message]

        output_messages = []
        fragment_loop_total_time = 0.0
        effective_logger.info(f"Starting fragmentation loop for {num_fragments} fragments.")

        for fragment_idx in range(num_fragments):
            loop_iter_start_time = time.perf_counter()
            effective_logger.debug(f"Fragment {fragment_idx + 1}/{num_fragments}: Starting processing.")

            page_calc_start_time = time.perf_counter()
            start_page = fragment_idx * pages_per_fragment
            end_page = min(start_page + pages_per_fragment, total_pages)
            page_calc_end_time = time.perf_counter()
            effective_logger.debug(
                f"  Page range calculation [{start_page}-{end_page-1}] "
                f"took: {page_calc_end_time - page_calc_start_time:.4f}s"
            )

            pdf_creation_start_time = time.perf_counter()
            new_pdf = pdfium.PdfDocument.new()
            pdf_creation_end_time = time.perf_counter()
            effective_logger.debug(
                f"  pdfium.PdfDocument.new() took: {pdf_creation_end_time - pdf_creation_start_time:.4f}s"
            )

            import_pages_start_time = time.perf_counter()
            new_pdf.import_pages(pdf, list(range(start_page, end_page)))
            import_pages_end_time = time.perf_counter()
            effective_logger.debug(
                f"  new_pdf.import_pages() took: {import_pages_end_time - import_pages_start_time:.4f}s "
                f"for {end_page - start_page} pages"
            )

            save_bytes_start_time = time.perf_counter()
            buffer = BytesIO()
            new_pdf.save(buffer)
            fragment_bytes = buffer.getvalue()
            save_bytes_end_time = time.perf_counter()
            effective_logger.debug(
                f"  new_pdf.save() to buffer took: {save_bytes_end_time - save_bytes_start_time:.4f}s"
            )

            base64_encode_start_time = time.perf_counter()
            fragment_base64 = base64.b64encode(fragment_bytes).decode("utf-8")
            base64_encode_end_time = time.perf_counter()
            effective_logger.debug(
                f"  base64.b64encode() took: {base64_encode_end_time - base64_encode_start_time:.4f}s"
            )

            close_pdf_start_time = time.perf_counter()
            new_pdf.close()
            close_pdf_end_time = time.perf_counter()
            effective_logger.debug(f"  new_pdf.close() took: {close_pdf_end_time - close_pdf_start_time:.4f}s")

            df_meta_copy_start_time = time.perf_counter()
            new_df = payload.copy(deep=True)
            new_metadata_dict = deepcopy(metadata)
            new_metadata_dict["content"] = fragment_base64
            new_df.at[0, "metadata"] = (
                json.dumps(new_metadata_dict) if isinstance(row["metadata"], str) else new_metadata_dict
            )
            df_meta_copy_end_time = time.perf_counter()
            effective_logger.debug(
                f"  DataFrame & metadata copy/update took: {df_meta_copy_end_time - df_meta_copy_start_time:.4f}s"
            )

            msg_copy_start_time = time.perf_counter()
            new_message = deepcopy(message)
            new_message.payload(new_df)
            msg_copy_end_time = time.perf_counter()
            effective_logger.debug(
                f"  IngestControlMessage deepcopy & payload set took: {msg_copy_end_time - msg_copy_start_time:.4f}s"
            )

            output_messages.append(new_message)
            loop_iter_end_time = time.perf_counter()
            iter_duration = loop_iter_end_time - loop_iter_start_time
            fragment_loop_total_time += iter_duration
            effective_logger.debug(
                f"Fragment {fragment_idx + 1}/{num_fragments}: Processing took: {iter_duration:.4f}s"
            )

        pdf_main_close_start_time = time.perf_counter()
        pdf.close()
        pdf_main_close_end_time = time.perf_counter()
        effective_logger.info(f"Main PDF close took: {pdf_main_close_end_time - pdf_main_close_start_time:.4f}s")
        effective_logger.info(
            f"Total time spent in fragmentation loop: {fragment_loop_total_time:.4f} seconds for {num_fragments} "
            f"fragments."
        )

        overall_end_time_frag = time.perf_counter()
        effective_logger.info(
            "_fragment_pdf_configured_internal: Total time (with fragmentation):"
            f" {overall_end_time_frag - overall_start_time:.4f} seconds"
        )
        return output_messages

    return _fragment_pdf_configured_internal
