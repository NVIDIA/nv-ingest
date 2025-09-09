# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
from minio import Minio

from nv_ingest_api.internal.enums.common import ContentTypeEnum

logger = logging.getLogger(__name__)

# TODO: Move these into microservice_entrypoint.py to populate the stage and validate them using the pydantic schema
# on startup.
_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_READ_ADDRESS = os.environ.get("MINIO_PUBLIC_ADDRESS", "http://minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")

# Parameters excluded from fsspec filesystem initialization
# These are nv-ingest specific configuration that shouldn't be passed to fsspec
_FSSPEC_EXCLUDED_PARAMS = [
    "content_types",
    "method",
    "collection_name",
    "bucket_name",
    "path",
    "bucket",
    "region",
    "path_prefix",
]

# AWS S3 credential parameters that fsspec accepts (ONLY credentials, no config)
_S3_CREDENTIAL_PARAMS = ["key", "secret", "token", "anon", "profile"]

# Map storage method to its credential parameters
_METHOD_CREDENTIAL_PARAMS = {
    "s3": _S3_CREDENTIAL_PARAMS,
}


def _ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    """
    Ensure that the specified bucket exists in MinIO, and create it if it does not.

    Parameters
    ----------
    client : Minio
        An instance of the Minio client.
    bucket_name : str
        The name of the bucket to check or create.
    """
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        logger.debug("Created bucket %s", bucket_name)
    else:
        logger.debug("Bucket %s already exists", bucket_name)


def _validate_row_metadata(idx: int, row: pd.Series, content_types: Dict) -> Optional[Tuple[str, Dict, Dict, str]]:
    """
    Validate row metadata and extract required fields for image upload.

    Parameters
    ----------
    idx : int
        Row index for error logging
    row : pd.Series
        DataFrame row containing document_type and metadata
    content_types : Dict
        Dictionary mapping document types to booleans

    Returns
    -------
    Optional[Tuple[str, Dict, Dict, str]]
        Tuple of (doc_type, metadata, source_metadata, source_id) if valid, None if invalid
    """
    doc_type = row.get("document_type")
    if doc_type not in content_types:
        return None

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        logger.error("Row %s: 'metadata' is not a dictionary", idx)
        return None

    # Validate required metadata fields
    if "content" not in metadata:
        logger.error("Row %s: missing 'content' in metadata", idx)
        return None

    if "source_metadata" not in metadata or not isinstance(metadata["source_metadata"], dict):
        logger.error("Row %s: missing or invalid 'source_metadata' in metadata", idx)
        return None

    source_metadata = metadata["source_metadata"]
    if "source_id" not in source_metadata:
        logger.error("Row %s: missing 'source_id' in source_metadata", idx)
        return None

    source_id = source_metadata["source_id"]
    return doc_type, metadata, source_metadata, source_id


def _upload_images_fsspec(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Upload images using fsspec for various storage backends (S3, GCS, Azure, local file).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing rows with content and associated metadata.
    params : Dict[str, Any]
        Configuration parameters including method type and credentials.
        Expected keys:
            - "method": Storage method type (currently supports: s3)
            - "content_types": Dict mapping document types to booleans.
            - "bucket": S3 bucket name (required for S3)
            - "region": S3 region (optional, uses default if not specified)
            - "path_prefix": Optional folder prefix for organizing files within the bucket
            - Credential parameters: "key", "secret", "token" (optional - uses automatic detection if not provided)

    File Structure & Organization
    -----------------------------
    Without path_prefix:
        bucket/document1.pdf/0.png
        bucket/document1.pdf/1.png
        bucket/document2.pdf/0.png

    With path_prefix="experiments/test-run-001":
        bucket/experiments/test-run-001/document1.pdf/0.png
        bucket/experiments/test-run-001/document1.pdf/1.png
        bucket/experiments/test-run-001/document2.pdf/0.png

    i.e.  "datasets/{dataset_name}/run-{id}"   # Dataset-based organization

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with metadata reflecting uploaded URLs.

    Raises
    ------
    ValueError
        If required configuration parameters are missing or invalid.
    """
    import fsspec

    # Validate required configuration
    content_types = params.get("content_types")
    if not isinstance(content_types, dict):
        raise ValueError("Invalid configuration: 'content_types' must be provided as a dictionary in params")

    method = params.get("method")
    if not method:
        raise ValueError("Storage method must be specified for fsspec storage")

    # Get credential parameters for this storage method
    allowed_credential_params = _METHOD_CREDENTIAL_PARAMS.get(method, [])

    # Only pass explicitly provided credential parameters to fsspec
    # This allows fsspec's automatic credential detection to work when no explicit creds are provided
    fs_params = {}
    provided_creds = []

    for param_name, param_value in params.items():
        # Skip nv-ingest specific parameters
        if param_name in _FSSPEC_EXCLUDED_PARAMS:
            continue

        # Only include credential parameters that are explicitly provided (not None/empty)
        if param_name in allowed_credential_params and param_value is not None and param_value != "":
            fs_params[param_name] = param_value
            provided_creds.append(param_name)

    # Log credential resolution method for debugging
    if provided_creds:
        logger.info("Using explicit credentials for %s storage: %s", method, provided_creds)
    else:
        logger.info(
            "No explicit credentials provided for %s storage - using automatic credential detection",
            method,
        )

    # Initialize fsspec filesystem
    try:
        fs = fsspec.filesystem(method, **fs_params)
        logger.debug("Successfully initialized %s filesystem with fsspec", method)
    except Exception as e:
        logger.error(
            "Failed to create fsspec filesystem for method %s with params %s: %s", method, list(fs_params.keys()), e
        )
        raise ValueError(f"Failed to initialize {method} filesystem: {e}") from e

    # Determine storage path/bucket
    storage_path = params.get("bucket", params.get("bucket_name", params.get("path", "")))

    # Get configurable path prefix for organizing files in folders
    path_prefix = params.get("path_prefix", "")
    if path_prefix and not path_prefix.endswith("/"):
        path_prefix = path_prefix + "/"

    # Process each row and upload images
    for idx, row in df.iterrows():
        try:
            # Validate row metadata using shared validation function
            validation_result = _validate_row_metadata(idx, row, content_types)
            if validation_result is None:
                continue

            doc_type, metadata, source_metadata, source_id = validation_result

            # Decode the content from base64
            content = base64.b64decode(metadata["content"].encode())

            # Determine image type (inline simple logic)
            image_type = "png"
            if doc_type == ContentTypeEnum.IMAGE:
                image_metadata = metadata.get("image_metadata", {})
                image_type = image_metadata.get("image_type", "png")

            # Construct destination file path with configurable folder structure
            # Structure: bucket/[path_prefix/]source_filename/index.image_type
            clean_path = os.path.basename(source_id)
            destination_file = f"{path_prefix}{clean_path}/{idx}.{image_type}"

            if storage_path:
                full_path = f"{storage_path}/{destination_file}"
            else:
                full_path = destination_file

            # Ensure directory exists for file systems that need it
            if method == "file":
                dir_path = os.path.dirname(full_path)
                if dir_path and not fs.exists(dir_path):
                    fs.makedirs(dir_path, exist_ok=True)

            # Upload using fsspec - use proper API that works across all filesystem types
            with fs.open(full_path, "wb") as f:
                f.write(content)

            # Generate public URL using fsspec
            try:
                public_url = fs.url(full_path)
            except (AttributeError, NotImplementedError):
                # Fallback for methods that don't support url() method
                if method == "file":
                    public_url = f"file://{full_path}"
                elif method == "s3":
                    endpoint = params.get("endpoint_url", "")
                    if endpoint:
                        public_url = f"{endpoint.rstrip('/')}/{full_path}"
                    else:
                        public_url = f"https://{storage_path}.s3.amazonaws.com/{destination_file}"
                else:
                    public_url = f"{method}://{full_path}"

            # Update metadata (inline simple logic)
            source_metadata["source_location"] = public_url

            if doc_type == ContentTypeEnum.IMAGE:
                logger.debug("Stored image data via fsspec (%s) for row %s: %s", method, idx, public_url)
                image_metadata = metadata.get("image_metadata", {})
                image_metadata["uploaded_image_url"] = public_url
                metadata["image_metadata"] = image_metadata
            elif doc_type == ContentTypeEnum.STRUCTURED:
                logger.debug("Stored structured image data via fsspec (%s) for row %s: %s", method, idx, public_url)
                table_metadata = metadata.get("table_metadata", {})
                table_metadata["uploaded_image_url"] = public_url
                metadata["table_metadata"] = table_metadata

            df.at[idx, "metadata"] = metadata

        except Exception as e:
            logger.exception("Failed to process row %s with fsspec method %s: %s", idx, method, e)
            # Continue processing remaining rows

    return df


def _upload_images_to_minio(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifies content within a DataFrame and uploads it to MinIO, updating the metadata with the uploaded URL.

    This function iterates over rows of the provided DataFrame. For rows whose "document_type" is listed
    in the provided 'content_types' configuration, it decodes the base64-encoded content, uploads the object to
    MinIO, and updates the metadata with the public URL. Errors during individual row processing are logged and
    skipped, so the process continues for remaining rows.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing rows with content and associated metadata.
    params : Dict[str, Any]
        A flat dictionary of configuration parameters for the upload. Expected keys include:
            - "content_types": Dict mapping document types to booleans.
            - "endpoint": URL for the MinIO service (optional; defaults to _DEFAULT_ENDPOINT).
            - "bucket_name": Bucket name for storing objects (optional; defaults to _DEFAULT_BUCKET_NAME).
            - "access_key": Access key for MinIO.
            - "secret_key": Secret key for MinIO.
            - "session_token": Session token for MinIO (optional).
            - "secure": Boolean indicating if HTTPS should be used.
            - "region": Region for the MinIO service (optional).

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

    endpoint: str = params.get("endpoint", _DEFAULT_ENDPOINT)
    bucket_name: str = params.get("bucket_name", _DEFAULT_BUCKET_NAME)

    # Initialize MinIO client
    client = Minio(
        endpoint,
        access_key=params.get("access_key"),
        secret_key=params.get("secret_key"),
        session_token=params.get("session_token"),
        secure=params.get("secure", False),
        region=params.get("region"),
    )

    # Ensure the bucket exists
    _ensure_bucket_exists(client, bucket_name)

    # Process each row and attempt to upload images
    for idx, row in df.iterrows():
        try:
            # Validate row metadata using shared validation function
            validation_result = _validate_row_metadata(idx, row, content_types)
            if validation_result is None:
                continue

            doc_type, metadata, source_metadata, source_id = validation_result

            # Decode the content from base64
            content = base64.b64decode(metadata["content"].encode())

            # Determine image type (inline simple logic)
            image_type = "png"
            if doc_type == ContentTypeEnum.IMAGE:
                image_metadata = metadata.get("image_metadata", {})
                image_type = image_metadata.get("image_type", "png")

            # Construct destination file path with URL encoding for MinIO
            encoded_source_id = quote(source_id, safe="")
            encoded_image_type = quote(image_type, safe="")
            destination_file = f"{encoded_source_id}/{idx}.{encoded_image_type}"

            # Upload the object to MinIO
            source_file = BytesIO(content)
            client.put_object(
                bucket_name,
                destination_file,
                source_file,
                length=len(content),
            )

            # Construct the public URL
            public_url = f"{_DEFAULT_READ_ADDRESS}/{bucket_name}/{destination_file}"

            # Update metadata (inline simple logic)
            source_metadata["source_location"] = public_url

            if doc_type == ContentTypeEnum.IMAGE:
                logger.debug("Stored image data to MinIO for row %s: %s", idx, public_url)
                image_metadata = metadata.get("image_metadata", {})
                image_metadata["uploaded_image_url"] = public_url
                metadata["image_metadata"] = image_metadata
            elif doc_type == ContentTypeEnum.STRUCTURED:
                logger.debug("Stored structured image data to MinIO for row %s: %s", idx, public_url)
                table_metadata = metadata.get("table_metadata", {})
                table_metadata["uploaded_image_url"] = public_url
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
    Processes a storage ledger DataFrame to upload images (and structured content) to MinIO.

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
        "content_types" (a dict mapping document types to booleans) along with connection and credential details.
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

    # Route to appropriate storage method
    method = task_config.get("method")
    if method == "minio":
        # Use existing MinIO implementation for backward compatibility
        result, execution_trace_log = _upload_images_to_minio(df_storage_ledger, task_config), {}
    elif method in ["s3", "gcs", "azure", "file"]:
        # Use fsspec for new storage methods
        result, execution_trace_log = _upload_images_fsspec(df_storage_ledger, task_config), {}
    else:
        raise ValueError(f"Unsupported storage method: {method}")
    _ = execution_trace_log

    return result
