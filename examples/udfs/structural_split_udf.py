#!/usr/bin/env python3
"""
Structural Text Splitter UDF for NV-Ingest Pipeline

This UDF splits text content by markdown headers while preserving document hierarchy.
All logic is self-contained with no external dependencies beyond the core API.

"""

import copy
import logging
import re
import uuid
from typing import List, Dict, Any


def structural_split(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    UDF function that splits text content by markdown headers.

    This function:
    1. Gets the DataFrame payload from the control message
    2. Finds rows with text content that have markdown headers
    3. Splits those rows into multiple chunks based on headers
    4. Creates new rows for each chunk with proper metadata
    5. Returns the updated control message

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing the payload and metadata

    Returns
    -------
    IngestControlMessage
        The modified control message with structurally split text
    """
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.info("UDF: Starting structural text splitting")

    # Get the payload DataFrame
    df = control_message.payload()
    if df is None or len(df) == 0:
        logger.warning("UDF: No payload found in control message")
        return control_message

    logger.info(f"UDF: Processing DataFrame with {len(df)} rows")

    # Configuration - which headers to split on
    markdown_headers = ["#", "##", "###", "####", "#####", "######"]

    # Find rows that should be split (text primitives regardless of original source type)
    rows_to_split = []
    rows_to_keep = []

    for idx, row in df.iterrows():
        # Check if this is a text primitive (regardless of whether it came from PDF, DOCX, etc.)
        is_text_primitive = row.get("document_type") == "text" or str(row.get("document_type", "")).lower() == "text"

        if is_text_primitive:
            rows_to_split.append((idx, row))
        else:
            rows_to_keep.append(row)

    logger.debug(
        f"UDF: Found {len(rows_to_split)} text primitives to split, {len(rows_to_keep)} non-text primitives to keep"
    )

    # Split the eligible rows
    new_rows = []
    for idx, row in rows_to_split:
        content = ""
        if isinstance(row.get("metadata"), dict):
            content = row["metadata"].get("content", "")

        # Try to decode base64 content if needed
        try:
            import base64

            decoded_content = base64.b64decode(content).decode("utf-8")
            content = decoded_content
            logger.debug("UDF: Decoded base64 content")
        except Exception:
            # Content is already plain text, use as-is
            pass

        if not content or not content.strip():
            # No content to split, keep original row
            new_rows.append(row.to_dict())
            continue

        # Split the content by markdown headers
        chunks = _split_by_markdown_headers(content, markdown_headers)

        if len(chunks) <= 1:
            # No splitting occurred, keep original row
            new_rows.append(row.to_dict())
        else:
            # Create new rows for each chunk
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                # Create new row based on original
                new_row = row.to_dict().copy()

                # Deep copy metadata to avoid reference issues
                metadata = copy.deepcopy(row.get("metadata", {}))
                metadata["content"] = chunk_text

                # Add chunk metadata
                if "custom_content" not in metadata:
                    metadata["custom_content"] = {}

                metadata["custom_content"]["chunk_index"] = i
                metadata["custom_content"]["total_chunks"] = len(chunks)
                metadata["custom_content"]["splitting_method"] = "structural_markdown"

                # Extract header info from chunk
                lines = chunk_text.split("\n")
                header_info = _extract_header_info(lines, markdown_headers)
                metadata["custom_content"].update(header_info)

                # Update row
                new_row["metadata"] = metadata
                new_row["uuid"] = str(uuid.uuid4())

                new_rows.append(new_row)

    # Combine split rows with unsplit rows
    all_rows = new_rows + [row.to_dict() for row in rows_to_keep]

    # Create new DataFrame
    new_df = pd.DataFrame(all_rows)

    # Update the control message
    control_message.payload(new_df)

    logger.info(f"UDF: Structural splitting complete: {len(df)} → {len(new_df)} rows")
    if len(new_df) > len(df):
        chunks_created = len(new_df) - len(df)
        logger.info(f"UDF: Created {chunks_created} additional chunks")

    return control_message


def _split_by_markdown_headers(text: str, headers: List[str]) -> List[str]:
    """Split content by markdown headers."""
    if not text or not text.strip():
        return [text]

    lines = text.split("\n")
    chunks = []
    current_chunk_lines = []

    # Create pattern to match any of the specified headers
    header_pattern = r"^(" + "|".join(re.escape(h) for h in headers) + r")\s+(.+)$"

    for line in lines:
        if re.match(header_pattern, line.strip()):
            # Found a header - finalize current chunk if it exists
            if current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines).strip()
                if chunk_content:
                    chunks.append(chunk_content)
                current_chunk_lines = []
            # Start new chunk with this header
            current_chunk_lines.append(line)
        else:
            # Regular content line
            current_chunk_lines.append(line)

    # Add final chunk
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines).strip()
        if chunk_content:
            chunks.append(chunk_content)

    return chunks if chunks else [text]


def _extract_header_info(lines: List[str], headers: List[str]) -> Dict[str, Any]:
    """Extract hierarchical header information from chunk lines."""
    header_pattern = r"^(" + "|".join(re.escape(h) for h in headers) + r")\s+(.+)$"

    # Find the first header in the chunk
    for line in lines:
        header_match = re.match(header_pattern, line.strip())
        if header_match:
            header_prefix = header_match.group(1)
            header_text = header_match.group(2)
            return {
                "hierarchical_header": f"{header_prefix} {header_text}",
                "header_level": len(header_prefix),
                "parent_headers": [],  # Simplified for now
            }

    # No header found
    return {"hierarchical_header": "(no headers found)", "header_level": 0, "parent_headers": []}


def structural_split_coarse(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    Alternative UDF that only splits on major headers (# and ##) for larger chunks.
    """
    logger = logging.getLogger(__name__)
    logger.info("UDF: Starting coarse structural text splitting")

    # For this example, we'll just call the main function
    # In practice, you would implement the full logic with different header patterns
    return structural_split(control_message)
