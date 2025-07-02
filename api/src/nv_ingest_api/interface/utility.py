# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO

import pandas as pd
from datetime import datetime
from typing import List, Union

from nv_ingest_api.internal.enums.common import ContentTypeEnum, DocumentTypeEnum

# ------------------------------------------------------------------------------
# Mapping from DocumentTypeEnum to ContentTypeEnum
# ------------------------------------------------------------------------------
DOCUMENT_TO_CONTENT_MAPPING = {
    DocumentTypeEnum.BMP: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.DOCX: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.HTML: ContentTypeEnum.TEXT,
    DocumentTypeEnum.JPEG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.PDF: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.PNG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.PPTX: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.SVG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.TIFF: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.TXT: ContentTypeEnum.TEXT,
    DocumentTypeEnum.MD: ContentTypeEnum.TEXT,
    DocumentTypeEnum.MP3: ContentTypeEnum.AUDIO,
    DocumentTypeEnum.WAV: ContentTypeEnum.AUDIO,
    DocumentTypeEnum.UNKNOWN: ContentTypeEnum.UNKNOWN,
}


# ------------------------------------------------------------------------------
# Helper function to read a file and return its base64-encoded string.
# ------------------------------------------------------------------------------
def read_file_as_base64(file_path: str) -> str:
    """
    Reads the file at file_path in binary mode and returns its base64-encoded string.
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return base64.b64encode(file_bytes).decode("utf-8")


# ------------------------------------------------------------------------------
# Helper function to read a BytesIO object and return its base64-encoded string.
# ------------------------------------------------------------------------------
def read_bytesio_as_base64(file_io: BytesIO) -> str:
    """
    Reads a BytesIO object and returns its base64-encoded string.

    Parameters:
        file_io (BytesIO): A file-like object containing binary data.

    Returns:
        str: The base64-encoded string representation of the file's contents.
    """
    file_bytes = file_io.getvalue()
    return base64.b64encode(file_bytes).decode("utf-8")


# ------------------------------------------------------------------------------
# Helper function to create source metadata.
# ------------------------------------------------------------------------------
def create_source_metadata(source_name: str, source_id: str, document_type: str) -> dict:
    """
    Creates a source metadata dictionary for a file.

    The source_type is set to the provided document_type.
    The date_created and last_modified fields are set to the current ISO timestamp.
    """
    now_iso = datetime.now().isoformat()
    return {
        "source_name": source_name,
        "source_id": source_id,
        "source_location": "",
        "source_type": document_type,  # e.g., "pdf", "png", etc.
        "collection_id": "",
        "date_created": now_iso,
        "last_modified": now_iso,
        "summary": "",
        "partition_id": -1,
        "access_level": "unknown",  # You may wish to adjust this if needed.
        "custom_content": {},
    }


# ------------------------------------------------------------------------------
# Helper function to create content metadata.
# ------------------------------------------------------------------------------
def create_content_metadata(document_type: str) -> dict:
    """
    Creates a content metadata dictionary for a file based on its document type.

    It maps the document type to the corresponding content type.
    """
    # Use the mapping; if document_type is not found, fallback to "unknown".
    content_type = DOCUMENT_TO_CONTENT_MAPPING.get(document_type, ContentTypeEnum.UNKNOWN)
    return {
        "type": content_type,
        "description": "",
        "page_number": -1,
        "hierarchy": {
            "page_count": -1,
            "page": -1,
            "block": -1,
            "line": -1,
            "span": -1,
            "nearby_objects": {
                "text": {"content": [], "bbox": [], "type": []},
                "images": {"content": [], "bbox": [], "type": []},
                "structured": {"content": [], "bbox": [], "type": []},
            },
        },
        "subtype": "",
        "custom_content": {},
    }


# ------------------------------------------------------------------------------
# Main helper function to build a DataFrame from lists of files.
# ------------------------------------------------------------------------------
def build_dataframe_from_files(
    file_paths: List[Union[str, BytesIO]],
    source_names: List[str],
    source_ids: List[str],
    document_types: List[str],
) -> pd.DataFrame:
    """
    Given lists of file paths (or BytesIO objects), source names, source IDs, and document types,
    reads each file (base64-encoding its contents) and constructs a DataFrame.

    For image content, 'image_metadata' is initialized as an empty dict, so it can later be updated.
    """
    rows = []
    # Validate that all lists have the same length.
    if not (len(file_paths) == len(source_names) == len(source_ids) == len(document_types)):
        raise ValueError("All input lists must have the same length.")

    for fp, sname, sid, d_type in zip(file_paths, source_names, source_ids, document_types):
        # Determine if fp is a file path (str) or a file-like object (e.g., BytesIO).
        if isinstance(fp, str):
            encoded_content = read_file_as_base64(fp)
        elif hasattr(fp, "read"):
            encoded_content = read_bytesio_as_base64(fp)
        else:
            raise ValueError("Each element in file_paths must be a string or a file-like object.")

        # Build metadata components.
        source_meta = create_source_metadata(sname, sid, d_type)
        content_meta = create_content_metadata(d_type)
        # If the content type is image, initialize image_metadata as {}.
        image_metadata = {} if content_meta.get("type") == ContentTypeEnum.IMAGE else None

        # Assemble the complete metadata dictionary.
        metadata = {
            "content": encoded_content,
            "content_url": "",
            "embedding": None,
            "source_metadata": source_meta,
            "content_metadata": content_meta,
            "audio_metadata": None,
            "text_metadata": None,
            "image_metadata": image_metadata,
            "table_metadata": None,
            "chart_metadata": None,
            "error_metadata": None,
            "info_message_metadata": None,
            "debug_metadata": None,
            "raise_on_failure": False,
        }

        # Build the row dictionary.
        row = {
            "source_name": sname,
            "source_id": sid,
            "content": encoded_content,
            "document_type": d_type,
            "metadata": metadata,
        }
        rows.append(row)

    # Create and return the DataFrame.
    return pd.DataFrame(rows)
