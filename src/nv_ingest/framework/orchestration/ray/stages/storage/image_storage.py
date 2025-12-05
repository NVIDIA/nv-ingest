# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import pandas as pd
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


@ray.remote
class ImageStorageStage(RayActorStage):
    """
    A Ray actor stage that stores images or structured content using an fsspec-compatible backend and updates
    metadata with storage URLs.

    This stage uses the validated configuration (ImageStorageModuleSchema) to process and store the DataFrame
    payload and updates the control message accordingly.
    """

    def __init__(self, config: ImageStorageModuleSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("ImageStorageStage configuration validated successfully.")
        except Exception as e:
            logger.exception("Error validating image storage config")
            raise e

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["store"])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by storing images or structured content.

        Parameters
        ----------
        control_message : IngestControlMessage
            The incoming message containing the DataFrame payload.

        Returns
        -------
        IngestControlMessage
            The updated message with storage URLs and trace info added.
        """
        logger.info("ImageStorageStage.on_data: Starting storage operation.")

        # Extract DataFrame payload.
        df_payload = control_message.payload()
        logger.debug("ImageStorageStage: Extracted payload with %d rows.", len(df_payload))

        # Remove the "store" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "store")
        # logger.debug("ImageStorageStage: Task configuration extracted: %s", pprint.pformat(task_config))

        stage_defaults = {
            "structured": self.validated_config.structured,
            "images": self.validated_config.images,
            "storage_uri": self.validated_config.storage_uri,
            "storage_options": self.validated_config.storage_options,
            "public_base_url": self.validated_config.public_base_url,
        }

        store_structured: bool = task_config.get("structured", stage_defaults["structured"])
        store_unstructured: bool = task_config.get("images", stage_defaults["images"])

        content_types: Dict[Any, Any] = {}
        if store_structured:
            content_types[ContentTypeEnum.STRUCTURED] = store_structured

        if store_unstructured:
            content_types[ContentTypeEnum.IMAGE] = store_unstructured

        params: Dict[str, Any] = task_config.get("params", {})

        storage_uri = task_config.get("storage_uri") or params.get("storage_uri") or stage_defaults["storage_uri"]
        storage_options = {
            **(stage_defaults["storage_options"] or {}),
            **(task_config.get("storage_options") or {}),
            **params.get("storage_options", {}),
        }
        if "public_base_url" in task_config:
            public_base_url = task_config["public_base_url"]
        else:
            public_base_url = params.get("public_base_url", stage_defaults["public_base_url"])

        storage_options = self._inject_storage_defaults(storage_uri, storage_options)

        storage_params: Dict[str, Any] = {
            "content_types": content_types,
            "storage_uri": storage_uri,
            "storage_options": storage_options,
        }
        if public_base_url:
            storage_params["public_base_url"] = public_base_url

        logger.debug("Processing storage task with parameters: %s", storage_params)

        # Store images or structured content.
        df_storage_ledger: pd.DataFrame = store_images_to_minio_internal(
            df_storage_ledger=df_payload,
            task_config=storage_params,
            storage_config={},
            execution_trace_log=None,
        )

        logger.info("Image storage operation completed. Updated payload has %d rows.", len(df_storage_ledger))

        # Update the control message payload.
        control_message.payload(df_storage_ledger)

        return control_message

    @staticmethod
    def _inject_storage_defaults(storage_uri: str, storage_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Populate storage options for common backends (e.g., MinIO/S3) using environment defaults.
        """
        parsed_scheme = urlparse(storage_uri).scheme.lower()
        merged_options: Dict[str, Any] = {k: v for k, v in storage_options.items() if v is not None}

        if parsed_scheme not in {"s3", "s3a", "s3n"}:
            return merged_options

        def _set_if_absent(key: str, env_var: str) -> None:
            if key not in merged_options and env_var in os.environ:
                merged_options[key] = os.environ[env_var]

        _set_if_absent("key", "MINIO_ACCESS_KEY")
        _set_if_absent("secret", "MINIO_SECRET_KEY")
        if "token" not in merged_options and os.environ.get("MINIO_SESSION_TOKEN"):
            merged_options["token"] = os.environ["MINIO_SESSION_TOKEN"]

        client_kwargs = dict(merged_options.get("client_kwargs", {}))
        endpoint = os.environ.get("MINIO_INTERNAL_ADDRESS")
        if not endpoint:
            endpoint = "http://minio:9000"
        if endpoint and not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        client_kwargs.setdefault("endpoint_url", endpoint)
        region = os.environ.get("MINIO_REGION")
        if region:
            client_kwargs.setdefault("region_name", region)
        if client_kwargs:
            merged_options["client_kwargs"] = client_kwargs

        return merged_options
