# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Dict, Any, Optional

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
    A Ray actor stage that stores images or structured content in MinIO and updates metadata with storage URLs.

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
            "enable_minio": self.validated_config.enable_minio,
            "enable_local_disk": self.validated_config.enable_local_disk,
            "local_output_path": self.validated_config.local_output_path,
        }

        store_structured: bool = task_config.get("structured", stage_defaults["structured"])
        store_unstructured: bool = task_config.get("images", stage_defaults["images"])

        content_types: Dict[Any, Any] = {}
        if store_structured:
            content_types[ContentTypeEnum.STRUCTURED] = store_structured

        if store_unstructured:
            content_types[ContentTypeEnum.IMAGE] = store_unstructured

        params: Dict[str, Any] = {
            "enable_minio": stage_defaults["enable_minio"],
            "enable_local_disk": stage_defaults["enable_local_disk"],
            "local_output_path": stage_defaults["local_output_path"],
            **task_config.get("params", {}),
        }
        for key in ("enable_minio", "enable_local_disk", "local_output_path"):
            if key in task_config and task_config[key] is not None:
                params[key] = task_config[key]

        # Inject MinIO credentials from environment if not provided in task params
        # This allows stage to work without explicit credentials in every .store() call
        if params.get("enable_minio", True):
            if "access_key" not in params:
                params["access_key"] = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
            if "secret_key" not in params:
                params["secret_key"] = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

        params["content_types"] = content_types

        logger.debug(f"Processing storage task with parameters: {params}")

        # Store images or structured content.
        df_storage_ledger: pd.DataFrame = store_images_to_minio_internal(
            df_storage_ledger=df_payload,
            task_config=params,
            storage_config={},
            execution_trace_log=None,
        )

        logger.info("Image storage operation completed. Updated payload has %d rows.", len(df_storage_ledger))

        # Update the control message payload.
        control_message.payload(df_storage_ledger)

        return control_message
