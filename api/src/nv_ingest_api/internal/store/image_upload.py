# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from io import BytesIO
from typing import Any, List, Optional
from typing import Dict
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
                image_type = image_metadata.get("image_type", "png")

            # Construct destination file path
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
            source_metadata["source_location"] = public_url

            if doc_type == ContentTypeEnum.IMAGE:
                logger.debug("Storing image data to Minio for row %s", idx)
                image_metadata = metadata.get("image_metadata", {})
                image_metadata["uploaded_image_url"] = public_url
                metadata["image_metadata"] = image_metadata
            elif doc_type == ContentTypeEnum.STRUCTURED:
                logger.debug("Storing structured image data to Minio for row %s", idx)
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

    result, execution_trace_log = _upload_images_to_minio(df_storage_ledger, task_config), {}
    _ = execution_trace_log

    return result
