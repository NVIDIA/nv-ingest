# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.mutate.filter import filter_images_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


class PythonImageFilterStage(PythonStage):
    """
    A Python stage that filters images within a DataFrame payload.

    It expects an IngestControlMessage containing a DataFrame with image documents. It then:
      1. Removes the "filter" task from the message.
      2. Calls the image filtering logic (via filter_images_internal) using a validated configuration.
      3. Updates the message payload with the filtered DataFrame.
    """

    def __init__(self, config: ImageFilterSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("PythonImageFilterStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Image Filter config: {e}")
            raise

    @traceable("image_filter")
    @filter_by_task(required_tasks=[("filter", {})])
    @nv_ingest_node_failure_try_except(annotation_id="image_filter", raise_on_failure=False)
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by filtering images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image documents.

        Returns
        -------
        IngestControlMessage
            The updated message with filtered images.
        """
        self._logger.debug("PythonImageFilterStage.on_data: Starting image filtering process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "filter" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "filter")
        self._logger.debug("Extracted task config: %s", task_config)

        task_params: Dict[str, Any] = task_config.get("params", {})

        # Perform image filtering.
        new_df = filter_images_internal(
            df_ledger=df_ledger,
            task_config=task_params,
            mutate_config=self.validated_config,
            execution_trace_log=None,
        )
        self._logger.info("Image filtering completed. Resulting DataFrame has %d rows.", len(new_df))

        # Update the message payload with the filtered DataFrame.
        control_message.payload(new_df)

        # Update statistics
        self.stats["processed"] += 1

        self._logger.debug("PythonImageFilterStage.on_data: Image filtering completed.")
        return control_message
