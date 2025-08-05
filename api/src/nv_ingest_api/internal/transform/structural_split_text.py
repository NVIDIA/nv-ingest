# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core implementation for structural text splitting.

This module provides functionality to split text documents by markdown headers
while preserving document structure and hierarchy. Unlike token-based splitting,
this approach respects the logical organization of content.
"""

import copy
import logging
import re
import uuid
from typing import Dict, List, Any, Optional

import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_structural_text_splitter_schema import (
    StructuralTextSplitterSchema
)
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


def _build_split_documents_structural(row, chunks: List[str]) -> List[Dict[str, Any]]:
    """Build documents from markdown header chunks."""
    documents: List[Dict] = []

    for i, text in enumerate(chunks):
        if text is None or not text.strip():
            continue

        metadata = row.metadata if hasattr(row, "metadata") and isinstance(row.metadata, dict) else {}
        metadata = copy.deepcopy(metadata)
        metadata["content"] = text
        
        # Add hierarchical metadata to custom_content
        metadata["custom_content"] = metadata.get("custom_content", {})
        metadata["custom_content"]["chunk_index"] = i
        metadata["custom_content"]["total_chunks"] = len(chunks)
        metadata["custom_content"]["splitting_method"] = "structural_markdown"
        
        # Extract hierarchical header info from the chunk
        lines = text.split('\n')
        header_info = _extract_header_info(lines, row.get('markdown_headers', ["#", "##", "###"]))
        metadata["custom_content"].update(header_info)

        documents.append({
            "document_type": ContentTypeEnum.TEXT.value,
            "metadata": metadata,
            "uuid": str(uuid.uuid4())
        })

    return documents


def _split_by_markdown_headers(text: str, headers: List[str]) -> List[str]:
    """Split content by markdown headers."""
    if not text or not text.strip():
        return [text]
    
    lines = text.split('\n')
    chunks = []
    current_chunk_lines = []
    
    # Create pattern to match any of the specified headers
    header_pattern = r'^(' + '|'.join(re.escape(h) for h in headers) + r')\s+(.+)$'
    
    for line in lines:
        if re.match(header_pattern, line.strip()):
            # Found a header - finalize current chunk if it exists
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines).strip()
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
        chunk_content = '\n'.join(current_chunk_lines).strip()
        if chunk_content:
            chunks.append(chunk_content)
    
    return chunks if chunks else [text]


def _extract_header_info(lines: List[str], headers: List[str]) -> Dict[str, Any]:
    """Extract hierarchical header information from chunk lines."""
    header_pattern = r'^(' + '|'.join(re.escape(h) for h in headers) + r')\s+(.+)$'
    
    # Find the first header in the chunk
    for line in lines:
        header_match = re.match(header_pattern, line.strip())
        if header_match:
            header_prefix = header_match.group(1)
            header_text = header_match.group(2)
            return {
                "hierarchical_header": f"{header_prefix} {header_text}",
                "header_level": len(header_prefix),
                "parent_headers": []  # Simplified - could be enhanced later
            }
    
    # No header found
    return {
        "hierarchical_header": "(no headers found)",
        "header_level": 0,
        "parent_headers": []
    }


@unified_exception_handler
def transform_text_split_structural_internal(
    df_transform_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    transform_config: StructuralTextSplitterSchema,
    execution_trace_log: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Internal function to split text by markdown headers.

    This function extracts text from documents that match a filter criteria based on source types,
    splits the text into chunks using markdown headers, and rebuilds document records with the
    split text. The resulting DataFrame contains both split and unsplit documents.

    Parameters
    ----------
    df_transform_ledger : pd.DataFrame
        DataFrame containing documents to be processed.
    task_config : Dict[str, Any]
        Task-specific configuration overrides.
    transform_config : StructuralTextSplitterSchema
        Configuration object providing default values for structural splitting.
    execution_trace_log : Optional[Dict[str, Any]]
        Optional execution trace for debugging (not used in this implementation).

    Returns
    -------
    pd.DataFrame
        DataFrame with processed documents. Documents with text matching the filter are split into chunks,
        and then merged with those that do not match the filter.
    """
    _ = execution_trace_log  # Placeholder for potential execution trace logging.

    # Override parameters using task_config, with fallback to transform_config.
    markdown_headers: List[str] = task_config.get("markdown_headers_to_split_on", transform_config.markdown_headers_to_split_on)
    split_source_types: List[str] = task_config.get("split_source_types", ["text"])

    logger.debug(
        f"Splitting text with headers: {markdown_headers}, "
        f"target source types: {split_source_types}"
    )

    # Filter to documents with text content (following exact pattern from split_text.py)
    # Handle both enum and string values for document_type
    text_type_condition = (
        df_transform_ledger["document_type"].isin([ContentTypeEnum.TEXT, "text"]) |
        df_transform_ledger["document_type"].astype(str).isin(["text", "ContentTypeEnum.TEXT"])
    )

    normalized_meta_df = pd.json_normalize(df_transform_ledger["metadata"], errors="ignore")
    if "source_metadata.source_type" in normalized_meta_df.columns:
        # Handle both enum and string values for source_type
        source_types_in_df = normalized_meta_df["source_metadata.source_type"]
        # Convert enum values to strings for comparison
        source_types_str = [
            str(val.value) if hasattr(val, 'value') else str(val) 
            for val in source_types_in_df
        ]
        source_type_condition = pd.Series(source_types_str).isin(split_source_types)
    else:
        source_type_condition = False

    bool_index = text_type_condition & source_type_condition
    df_filtered: pd.DataFrame = df_transform_ledger.loc[bool_index]

    logger.debug(
        f"Filtering results: text_type_condition={text_type_condition.sum()}, "
        f"source_type_condition={source_type_condition.sum() if hasattr(source_type_condition, 'sum') else source_type_condition}, "
        f"final_filtered={len(df_filtered)}"
    )

    if df_filtered.empty:
        logger.debug("No documents matched filtering criteria, returning original DataFrame")
        return df_transform_ledger

    split_docs: List[Dict[str, Any]] = []
    for _, row in df_filtered.iterrows():
        content: str = row["metadata"]["content"] if row["metadata"]["content"] is not None else ""
        
        # Decode base64 content if needed
        try:
            import base64
            decoded_content = base64.b64decode(content).decode('utf-8')
            logger.debug(f"Successfully decoded base64 content: {decoded_content[:100]}...")
            content = decoded_content
        except Exception as e:
            logger.debug(f"Content is not base64 encoded, using as-is: {e}")
            # Content is already plain text, use as-is
        
        # Store headers in row for use in _build_split_documents_structural
        row_with_headers = row.copy()
        row_with_headers['markdown_headers'] = markdown_headers
        chunks: List[str] = _split_by_markdown_headers(content, markdown_headers)
        split_docs.extend(_build_split_documents_structural(row_with_headers, chunks))

    split_docs_df: pd.DataFrame = pd.DataFrame(split_docs)

    # Merge split documents with unsplit documents (following exact pattern from split_text.py)
    merged_df: pd.DataFrame = pd.concat([split_docs_df, df_transform_ledger[~bool_index]], axis=0).reset_index(
        drop=True
    )

    return merged_df 