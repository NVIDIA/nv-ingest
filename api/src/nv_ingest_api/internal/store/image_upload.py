# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from upath import UPath

from nv_ingest_api.internal.enums.common import ContentTypeEnum

logger = logging.getLogger(__name__)


def _resolve_storage_root(storage_uri: str, storage_options: Dict[str, Any]) -> Tuple[UPath, str]:
    """
    Construct a UPath instance rooted at the configured URI and return both the root path and protocol.
    """
    storage_root = UPath(storage_uri, **storage_options)
    protocol = storage_root._url.scheme or "file"
    return storage_root, protocol


def _extract_image_type(doc_type: Any, metadata: Dict[str, Any]) -> str:
    """
    Determine the image type to use when writing the decoded content based on the document type.
    """

    def _normalize(raw_value: Any, default: str = "png") -> str:
        if raw_value is None:
            return default
        if hasattr(raw_value, "value"):
            return str(raw_value.value).lower()
        string_value = str(raw_value).strip()
        return string_value.lower() if string_value else default

    if doc_type == ContentTypeEnum.IMAGE:
        image_metadata = metadata.get("image_metadata", {})
        return _normalize(image_metadata.get("image_type"))

    if doc_type == ContentTypeEnum.STRUCTURED:
        table_metadata = metadata.get("table_metadata", {})
        return _normalize(table_metadata.get("image_type"))

    return "png"


def _build_destination_path(storage_root: UPath, source_id: str, row_index: int, image_type: str) -> Tuple[UPath, str]:
    """
    Build the destination UPath for the decoded content and return both the destination and relative key.
    """
    safe_source_name = os.path.basename(source_id.rstrip("/")) or "source"
    clean_source_name = safe_source_name.replace("/", "_")

    destination: UPath = storage_root / clean_source_name / f"{row_index}.{image_type}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_key = destination.relative_to(storage_root).as_posix()
    return destination, relative_key


def _upload_images_via_fsspec(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifies content within a DataFrame and persists it using an fsspec-compatible filesystem, updating
    metadata with the resulting URIs.

    This function iterates over rows of the provided DataFrame. For rows whose "document_type" is listed
    in the provided 'content_types' configuration, it decodes the base64-encoded content, writes the object
    via fsspec/UPath, and updates the metadata with the resolved URL. Errors during individual row processing
    are logged and skipped so the process continues for remaining rows.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing rows with content and associated metadata.
    params : Dict[str, Any]
        A flat dictionary of configuration parameters for the upload. Expected keys include:
            - "content_types": Dict mapping document types to booleans.
            - "storage_uri": Base URI (file://, s3://, etc.) where images should be written.
            - "storage_options": Optional dictionary forwarded to UPath/fsspec constructors.
            - "public_base_url": Optional HTTP(s) base used to surface stored objects.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with metadata reflecting the uploaded URLs. Rows that encountered errors
        during processing will remain unchanged.

    Raises
    ------
    ValueError
        If the required "content_types" key is missing or is not a dictionary.
    Exception
        Propagates any critical exceptions not handled at the row level.
    """
    # Validate required configuration
    content_types = params.get("content_types")
    if not isinstance(content_types, dict):
        raise ValueError("Invalid configuration: 'content_types' must be provided as a dictionary in params")

    storage_uri: Optional[str] = params.get("storage_uri")
    if not storage_uri or not storage_uri.strip():
        raise ValueError("`storage_uri` must be provided in task params.")

    storage_options: Dict[str, Any] = params.get("storage_options") or {}
    public_base_url: Optional[str] = params.get("public_base_url")

    storage_root, protocol = _resolve_storage_root(storage_uri, storage_options)

    # Process each row and attempt to upload images
    for idx, row in df.iterrows():
        try:
            doc_type = row.get("document_type")
            if doc_type not in content_types:
                continue

            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                logger.error("Row %s: 'metadata' is not a dictionary", idx)
                continue

            # Validate required metadata fields
            if "content" not in metadata:
                logger.error("Row %s: missing 'content' in metadata", idx)
                continue

            if "source_metadata" not in metadata or not isinstance(metadata["source_metadata"], dict):
                logger.error("Row %s: missing or invalid 'source_metadata' in metadata", idx)
                continue

            source_metadata = metadata["source_metadata"]
            if "source_id" not in source_metadata:
                logger.error("Row %s: missing 'source_id' in source_metadata", idx)
                continue

            # Decode the content from base64
            content = base64.b64decode(metadata["content"].encode())
            source_id = source_metadata["source_id"]

            image_type = _extract_image_type(doc_type, metadata)

            # Construct destination file path
            destination, relative_key = _build_destination_path(
                storage_root=storage_root,
                source_id=source_id,
                row_index=idx,
                image_type=image_type,
            )
            with destination.open("wb") as target_file:
                target_file.write(content)

            destination_uri = destination.as_uri()
            public_url: Optional[str] = None
            if public_base_url:
                public_url = f"{public_base_url.rstrip('/')}/{relative_key}"

            primary_uri = public_url or destination_uri
            source_metadata["source_location"] = primary_uri

            local_uri: Optional[str] = None
            if protocol == "file":
                local_uri = destination.path
                source_metadata["local_source_location"] = local_uri

            if doc_type == ContentTypeEnum.IMAGE:
                logger.debug("Persisting image data for row %s", idx)
                image_metadata = metadata.get("image_metadata", {})
                if public_url is not None:
                    image_metadata["uploaded_image_url"] = public_url
                if local_uri is not None:
                    image_metadata["uploaded_image_local_path"] = local_uri
                metadata["image_metadata"] = image_metadata
            elif doc_type == ContentTypeEnum.STRUCTURED:
                logger.debug("Persisting structured image data for row %s", idx)
                table_metadata = metadata.get("table_metadata", {})
                if public_url is not None:
                    table_metadata["uploaded_image_url"] = public_url
                if local_uri is not None:
                    table_metadata["uploaded_image_local_path"] = local_uri
                metadata["table_metadata"] = table_metadata

            df.at[idx, "metadata"] = metadata

        except Exception as e:
            logger.exception("Failed to process row %s: %s", idx, e)
            # Continue processing the remaining rows

    return df


def store_images_to_minio_internal(
    df_storage_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    storage_config: Dict[str, Any],
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Processes a storage ledger DataFrame to persist images (and structured content) via an fsspec-compatible
    filesystem.

    This function validates the input DataFrame and task configuration, then creates a mask to select rows
    where the "document_type" is among the desired types specified in the configuration. If matching rows are
    found, it calls the internal upload function to process and update the DataFrame; otherwise, it returns the
    original DataFrame unmodified.

    Parameters
    ----------
    df_storage_ledger : pd.DataFrame
        The DataFrame containing storage ledger information, which must include at least the columns
        "document_type" and "metadata".
    task_config : Dict[str, Any]
        A flat dictionary containing configuration parameters for image storage. Expected to include the key
        "content_types" (a dict mapping document types to booleans) along with `storage_uri`,
        `storage_options`, and optional presentation hints such as `public_base_url`.
    storage_config : Dict[str, Any]
        A dictionary reserved for additional storage configuration (currently unused).
    execution_trace_log : Optional[List[Any]], optional
        An optional list for capturing execution trace details (currently unused), by default None.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after attempting to upload images for rows with matching document types. Rows
        that do not match remain unchanged.

    Raises
    ------
    ValueError
        If the input DataFrame is missing required columns or if the task configuration is invalid.
    """
    # Validate that required keys and columns exist
    if "content_types" not in task_config or not isinstance(task_config["content_types"], dict):
        raise ValueError("Task configuration must include a valid 'content_types' dictionary.")

    if "document_type" not in df_storage_ledger.columns:
        raise ValueError("Input DataFrame must contain a 'document_type' column.")

    content_types = task_config["content_types"]

    # Create a mask for rows where "document_type" is one of the desired types
    storage_obj_mask = df_storage_ledger["document_type"].isin(list(content_types.keys()))
    if (~storage_obj_mask).all():
        logger.debug("No storage objects matching %s found in the DataFrame.", content_types)
        return df_storage_ledger

    result, execution_trace_log = _upload_images_via_fsspec(df_storage_ledger, task_config), {}
    _ = execution_trace_log

    return result
