# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
import posixpath
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fsspec.core import url_to_fs
from minio import Minio

from nv_ingest_api.internal.enums.common import ContentTypeEnum

logger = logging.getLogger(__name__)

# TODO: Move these into microservice_entrypoint.py to populate the stage and validate them using the pydantic schema
# on startup.
_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_READ_ADDRESS = os.environ.get("MINIO_PUBLIC_ADDRESS", "http://minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")


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
            - "enable_minio": Boolean that toggles MinIO uploads (default True).
            - "enable_local_disk": Boolean that toggles fsspec-backed disk persistence (default False).
            - "local_output_path": Root directory or URL where files should be written when local disk is enabled.
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

    enable_minio: bool = params.get("enable_minio", True)
    enable_local_disk: bool = params.get("enable_local_disk", False)
    local_output_path: Optional[str] = params.get("local_output_path")

    if not enable_minio and not enable_local_disk:
        raise ValueError("At least one storage backend must be enabled.")

    if enable_local_disk and not local_output_path:
        raise ValueError("`local_output_path` must be provided when enable_local_disk=True.")

    bucket_name: Optional[str] = None
    client: Optional[Minio] = None
    if enable_minio:
        endpoint: str = params.get("endpoint", _DEFAULT_ENDPOINT)
        bucket_name = params.get("bucket_name", _DEFAULT_BUCKET_NAME)

        # Initialize MinIO client
        # Credentials are injected by ImageStorageStage from environment
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

    fs = None
    fs_base_path: Optional[str] = None
    normalized_local_base: Optional[str] = None
    if enable_local_disk:
        fs, fs_base_path = url_to_fs(local_output_path)
        normalized_local_base = local_output_path.rstrip("/")

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

            # Determine image type (default to 'png')
            image_type = "png"
            if doc_type == ContentTypeEnum.IMAGE:
                image_metadata = metadata.get("image_metadata", {})
                raw_image_type = image_metadata.get("image_type", "png")
                # Handle both enum and string values
                if hasattr(raw_image_type, "value"):
                    # It's an enum, get the string value
                    image_type = raw_image_type.value.lower()
                else:
                    image_type = str(raw_image_type).lower()
            elif doc_type == ContentTypeEnum.STRUCTURED:
                # Structured content (tables/charts) may also have image_type
                table_metadata = metadata.get("table_metadata", {})
                raw_image_type = table_metadata.get("image_type", "png")
                # Handle both enum and string values
                if hasattr(raw_image_type, "value"):
                    image_type = raw_image_type.value.lower()
                else:
                    image_type = str(raw_image_type).lower()

            # Construct destination file path
            # Extract just the filename from source_id for cleaner organization
            clean_source_name = os.path.basename(source_id).replace("/", "_")

            # For MinIO: Use clean paths with forward slashes (S3 object keys support / as delimiter)
            minio_destination_file = f"{clean_source_name}/{idx}.{image_type}"

            # For local disk: Use the same clean filesystem paths
            local_destination_file = f"{clean_source_name}/{idx}.{image_type}"

            public_url: Optional[str] = None
            if enable_minio and client is not None and bucket_name is not None:
                # Upload the object to MinIO using URL-encoded path
                source_file = BytesIO(content)
                client.put_object(
                    bucket_name,
                    minio_destination_file,
                    source_file,
                    length=len(content),
                )

                # Construct the public URL
                public_url = f"{_DEFAULT_READ_ADDRESS}/{bucket_name}/{minio_destination_file}"
                source_metadata["source_location"] = public_url

            local_uri: Optional[str] = None
            if enable_local_disk and fs is not None and fs_base_path is not None and normalized_local_base is not None:
                # Use clean path for local filesystem
                disk_target_path = posixpath.join(fs_base_path.rstrip("/"), local_destination_file)
                fs.makedirs(posixpath.dirname(disk_target_path), exist_ok=True)
                with fs.open(disk_target_path, "wb") as local_file:
                    local_file.write(content)

                local_uri = f"{normalized_local_base}/{local_destination_file}"
                source_metadata["local_source_location"] = local_uri

                # When MinIO is disabled, fall back to local path for source_location.
                if not enable_minio:
                    source_metadata["source_location"] = local_uri

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
        Optional keys such as "enable_minio", "enable_local_disk", and "local_output_path" control which backends
        are used for persistence.
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

    result, execution_trace_log = _upload_images_to_minio(df_storage_ledger, task_config), {}
    _ = execution_trace_log

    return result
