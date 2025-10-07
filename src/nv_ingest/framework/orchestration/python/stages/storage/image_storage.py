# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


class PythonImageStorageStage(PythonStage):
    """
    A Python stage that stores images or structured content in MinIO and updates metadata with storage URLs.

    This stage uses the validated configuration (ImageStorageModuleSchema) to process and store the DataFrame
    payload and updates the control message accordingly.
    """

    def __init__(self, config: ImageStorageModuleSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("PythonImageStorageStage configuration validated successfully.")
        except Exception as e:
            logger.exception("Error validating image storage config")
            raise e

    @nv_ingest_node_failure_try_except(annotation_id="image_storage", raise_on_failure=False)
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["store"])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by storing images or structured content.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image or structured content data.

        Returns
        -------
        IngestControlMessage
            The updated message with storage metadata.
        """
        self._logger.debug("PythonImageStorageStage.on_data: Starting image storage process.")

        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_payload))

        # Remove the "store" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "store")
        self._logger.debug("Extracted task config: %s", task_config)

        params = task_config.get("params", {})
        self._logger.debug(f"Processing storage task with parameters: {params}")

        # Store images or structured content.
        df_storage_ledger: pd.DataFrame = store_images_to_minio_internal(
            df_storage_ledger=df_payload,
            task_config=params,
            storage_config={},
            execution_trace_log=None,
        )

        self._logger.info("Image storage operation completed. Updated payload has %d rows.", len(df_storage_ledger))

        # Update the control message payload.
        control_message.payload(df_storage_ledger)

        # Update statistics
        self.stats["processed"] += 1

        self._logger.debug("PythonImageStorageStage.on_data: Image storage completed.")
        return control_message
