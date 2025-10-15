# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for analyzing document-level chunk composition from nv-ingest results.

This module provides analysis capabilities for understanding the distribution and types
of extracted content elements across individual documents. It enables customers to
gain visibility into their document composition for performance optimization and
capacity planning decisions.
"""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def analyze_document_chunks(
    results: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Analyze ingestor results to count elements by type and page for each document.

    This function processes results from nv-ingest ingestion and provides a per-document,
    per-page breakdown of extracted content types, enabling customers to understand document
    composition and page-level distribution for optimization and planning purposes.

    Parameters
    ----------
    results : Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
        Ingestor results from ingestor.ingest() in standard List[List[Dict]] format,
        or flattened List[Dict] format. Handles both regular lists and
        LazyLoadedList objects automatically.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, int]]]
        Dictionary mapping document names to page-level element type counts with structure:
        {
            "document1.pdf": {
                "total": {
                    "text": 7, "charts": 1, "tables": 1,
                    "unstructured_images": 0, "infographics": 0, "page_images": 0
                },
                "1": {
                    "text": 3, "charts": 1, "tables": 0,
                    "unstructured_images": 0, "infographics": 0, "page_images": 0
                },
                "2": {
                    "text": 4, "charts": 0, "tables": 1,
                    "unstructured_images": 0, "infographics": 0, "page_images": 0
                }
            },
            "document2.pdf": {...}
        }

    Notes
    -----
    - Requires purge_results_after_upload=False in vdb_upload() configuration
    - Automatically handles LazyLoadedList objects from nv-ingest client
    - Returns zero counts for missing element types
    - Assumes valid nv-ingest output format with guaranteed metadata structure

    Examples
    --------
    >>> from nv_ingest_client.util.document_analysis import analyze_document_chunks
    >>>
    >>> # After running ingestion
    >>> results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    >>>
    >>> # Analyze document composition by page
    >>> breakdown = analyze_document_chunks(results)
    >>>
    >>> for doc_name, pages in breakdown.items():
    ...     total_counts = pages["total"]
    ...     total_elements = sum(total_counts.values())
    ...     page_count = len(pages) - 1  # Subtract 1 for "total" key
    ...     print(f"{doc_name}: {total_elements} elements across {page_count} pages")
    ...     print(f"  total: {total_elements} elements ({total_counts['text']} text, {total_counts['charts']} charts)")
    ...     for page_name, counts in pages.items():
    ...         if page_name != "total":  # Skip total when listing pages
    ...             page_total = sum(counts.values())
    ...             print(
                f"  page {page_name}: {page_total} elements "
                f"({counts['text']} text, {counts['charts']} charts)"
            )
    """

    if not results:
        logger.warning("No results provided for analysis")
        return {}

    # Normalize input format to handle both List[List[Dict]] and List[Dict] structures
    normalized_results = _normalize_results_format(results)

    # Group elements by document name and page number
    document_page_elements = defaultdict(lambda: defaultdict(list))

    for doc_results in normalized_results:
        # Handle LazyLoadedList and other iterable types
        elements = _extract_elements_from_doc(doc_results)

        for element in elements:
            doc_name = _extract_document_name(element)
            page_key = _extract_page_key(element)
            document_page_elements[doc_name][page_key].append(element)

    # Count element types per page within each document and calculate totals
    document_page_counts = {}

    for doc_name, pages in document_page_elements.items():
        document_page_counts[doc_name] = {}
        total_counts = _initialize_element_counts()

        for page_key, elements in pages.items():
            counts = _initialize_element_counts()

            for element in elements:
                element_type = _categorize_element(element)
                counts[element_type] += 1
                total_counts[element_type] += 1  # Add to document total

            document_page_counts[doc_name][page_key] = counts

        # Add the total counts for this document
        document_page_counts[doc_name]["total"] = total_counts

    if document_page_counts:
        total_docs = len(document_page_counts)
        total_pages = sum(len(pages) - 1 for pages in document_page_counts.values())  # Subtract 1 for "total" key
        total_elements = sum(sum(page_counts["total"].values()) for page_counts in document_page_counts.values())
        logger.info(f"Analyzed {total_elements} elements across {total_pages} pages in {total_docs} documents")
    else:
        logger.warning("No valid documents found for analysis")

    return document_page_counts


def _normalize_results_format(results: Union[List[List[Dict]], List[Dict]]) -> List[List[Dict]]:
    """
    Normalize various input formats to consistent List[List[Dict]] structure.

    Parameters
    ----------
    results : Union[List[List[Dict]], List[Dict]]
        Input results in various formats

    Returns
    -------
    List[List[Dict]]
        Normalized results in standard format
    """

    if not results:
        return []

    # Handle List[List[Dict]] or List[LazyLoadedList] formats
    if isinstance(results, list) and len(results) > 0:
        first_elem = results[0]
        # Check for list, LazyLoadedList, or any sequence-like object
        if isinstance(first_elem, list) or (
            hasattr(first_elem, "__iter__") and hasattr(first_elem, "__len__") and not isinstance(first_elem, dict)
        ):
            return results

    # Handle flattened List[Dict] format by grouping elements by document
    if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
        doc_groups = defaultdict(list)
        for element in results:
            doc_name = _extract_document_name(element)
            doc_groups[doc_name].append(element)

        return list(doc_groups.values())

    # Fallback for unexpected formats
    return [[item] for item in results if item]


def _extract_elements_from_doc(doc_results) -> List[Dict]:
    """
    Extract elements from document results, handling various data types.

    Parameters
    ----------
    doc_results : Any
        Document results which may be a list, LazyLoadedList, or other iterable

    Returns
    -------
    List[Dict]
        List of element dictionaries
    """

    if isinstance(doc_results, list):
        return doc_results
    elif hasattr(doc_results, "__iter__") and hasattr(doc_results, "__len__"):
        # Handle LazyLoadedList and other sequence-like objects
        return list(doc_results)
    else:
        # Single element case
        return [doc_results] if doc_results else []


def _extract_document_name(element: Dict[str, Any]) -> str:
    """
    Extract clean document name from element metadata.

    Parameters
    ----------
    element : Dict[str, Any]
        Element dictionary containing metadata

    Returns
    -------
    str
        Clean document filename (basename of source_id)
    """

    # nv-ingest guarantees this structure exists
    source_id = element["metadata"]["source_metadata"]["source_id"]
    return os.path.basename(source_id)


def _extract_page_key(element: Dict[str, Any]) -> str:
    """
    Extract page key from element metadata for consistent page naming.

    Parameters
    ----------
    element : Dict[str, Any]
        Element dictionary containing metadata

    Returns
    -------
    str
        Page number as string (e.g., "1", "2", or "unknown")
    """

    try:
        page_number = element["metadata"]["content_metadata"]["page_number"]
        if page_number is not None and page_number >= 0:
            return str(page_number)
        else:
            return "unknown"
    except (KeyError, TypeError):
        logger.warning("Missing or invalid page_number in element metadata")
        return "unknown"


def _categorize_element(element: Dict[str, Any]) -> str:
    """
    Categorize element by type using document_type and content metadata.

    Parameters
    ----------
    element : Dict[str, Any]
        Element dictionary with document_type and metadata fields

    Returns
    -------
    str
        Element category: "text", "charts", "tables", "unstructured_images",
        "infographics", or "page_images"
    """

    doc_type = element["document_type"]

    # Text elements
    if doc_type == "text":
        return "text"

    # Structured elements with subtypes
    elif doc_type == "structured":
        subtype = element["metadata"]["content_metadata"]["subtype"]
        if subtype == "chart":
            return "charts"
        elif subtype == "table":
            return "tables"
        elif subtype == "infographic":
            return "infographics"
        elif subtype == "page_image":
            return "page_images"

    # Image elements (unstructured)
    elif doc_type == "image":
        return "unstructured_images"

    # Should not reach here with valid nv-ingest output
    logger.warning(f"Unexpected element type: {doc_type}")
    return "text"  # Default to text for safety


def _initialize_element_counts() -> Dict[str, int]:
    """
    Initialize element counts dictionary with all supported types.

    Returns
    -------
    Dict[str, int]
        Dictionary with zero counts for all element types
    """

    return {
        "text": 0,
        "charts": 0,
        "tables": 0,
        "unstructured_images": 0,
        "infographics": 0,
        "page_images": 0,
    }
