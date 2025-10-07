# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


class PythonImageCaptionStage(PythonStage):
    """
    A Python stage that extracts image captions from a DataFrame payload.

    This stage validates its configuration (using ImageCaptionExtractionSchema), then processes the DataFrame
    via transform_image_create_vlm_caption_internal. The updated DataFrame and any extraction trace info
    are stored in the control message.
    """

    def __init__(self, config: ImageCaptionExtractionSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("PythonImageCaptionStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Image Caption config: {e}")
            raise

    @nv_ingest_node_failure_try_except(annotation_id="image_caption", raise_on_failure=False)
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=[("caption", {})])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting image captions.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image data.

        Returns
        -------
        IngestControlMessage
            The updated message with image captions.
        """
        self._logger.debug("PythonImageCaptionStage.on_data: Starting image caption extraction process.")

        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_payload))

        # Remove the "caption" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "caption")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform image caption extraction.
        new_df = transform_image_create_vlm_caption_internal(
            df_payload, task_config=task_config, transform_config=self.validated_config
        )
        self._logger.info("Image caption extraction completed. New payload has %d rows.", len(new_df))

        # Update the control message with the new DataFrame.
        control_message.payload(new_df)

        # Update statistics
        self.stats["processed"] += 1

        self._logger.info("PythonImageCaptionStage.on_data: Updated control message and returning.")
        return control_message
