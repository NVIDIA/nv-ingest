# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

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


def create_pdf_fragmenter(pages_per_fragment: int = 10, add_overlap: bool = False, overlap_pages: int = 1):
    """
    Factory function to create a PDF fragmenter with specific configuration using pypdfium2.

    Parameters
    ----------
    pages_per_fragment : int
        Number of pages per fragment
    add_overlap : bool
        Whether to add overlapping pages between fragments
    overlap_pages : int
        Number of pages to overlap (if add_overlap is True)

    Returns
    -------
    Callable[[IngestControlMessage], List[IngestControlMessage]]
        A configured PDF fragmenter function
    """

    def fragment_pdf_configured(message: IngestControlMessage) -> List[IngestControlMessage]:
        payload = message.payload()

        if len(payload) != 1:
            raise ValueError(f"Expected single row DataFrame, got {len(payload)} rows")

        row = payload.iloc[0]

        if row["document_type"] != "pdf":
            return [message]

        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        pdf_base64 = metadata["content"]
        pdf_bytes = base64.b64decode(pdf_base64)

        # Load PDF with pypdfium2
        pdf = pdfium.PdfDocument(pdf_bytes)
        total_pages = len(pdf)

        # Calculate fragments with overlap if requested
        if add_overlap:
            effective_pages_per_fragment = pages_per_fragment - overlap_pages
            num_fragments = (total_pages + effective_pages_per_fragment - 1) // effective_pages_per_fragment
        else:
            num_fragments = (total_pages + pages_per_fragment - 1) // pages_per_fragment

        if num_fragments == 1:
            pdf.close()
            return [message]

        output_messages = []

        for fragment_idx in range(num_fragments):
            if add_overlap and fragment_idx > 0:
                start_page = fragment_idx * (pages_per_fragment - overlap_pages)
            else:
                start_page = fragment_idx * pages_per_fragment

            end_page = min(start_page + pages_per_fragment, total_pages)

            # Create new PDF for fragment
            new_pdf = pdfium.PdfDocument.new()

            # Import pages (pypdfium2 uses 0-based indexing)
            new_pdf.import_pages(pdf, list(range(start_page, end_page)))

            # Save to bytes
            buffer = BytesIO()
            new_pdf.save(buffer)
            fragment_bytes = buffer.getvalue()
            fragment_base64 = base64.b64encode(fragment_bytes).decode("utf-8")

            # Close the fragment
            new_pdf.close()

            new_df = payload.copy(deep=True)
            new_metadata = deepcopy(metadata)
            new_metadata["content"] = fragment_base64
            new_metadata["fragment_info"] = {
                "fragment_index": fragment_idx,
                "total_fragments": num_fragments,
                "start_page": start_page,
                "end_page": end_page,
                "total_pages": total_pages,
                "pages_in_fragment": end_page - start_page,
                "has_overlap": add_overlap,
                "overlap_pages": overlap_pages if add_overlap else 0,
            }

            new_df.at[0, "metadata"] = json.dumps(new_metadata) if isinstance(row["metadata"], str) else new_metadata

            new_message = deepcopy(message)
            new_message.payload(new_df)

            output_messages.append(new_message)

        # Clean up
        pdf.close()

        return output_messages

    return fragment_pdf_configured
