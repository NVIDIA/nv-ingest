# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pprint
from typing import Any, Optional

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


@ray.remote
class ImageCaptionTransformStage(RayActorStage):
    """
    A Ray actor stage that extracts image captions from a DataFrame payload.

    This stage validates its configuration (using ImageCaptionExtractionSchema), then processes the DataFrame
    via transform_image_create_vlm_caption_internal. The updated DataFrame and any extraction trace info
    are stored in the control message.
    """

    def __init__(self, config: ImageCaptionExtractionSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("ImageCaptionTransformStage configuration validated.")
        except Exception as e:
            logger.exception("Error validating caption extraction config")
            raise e

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["caption"])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting image captions.

        Parameters
        ----------
        control_message : IngestControlMessage
            The incoming message containing the DataFrame payload.

        Returns
        -------
        IngestControlMessage
            The updated message with the extracted captions.
        """
        logger.info("ImageCaptionTransformStage.on_data: Starting image caption extraction.")

        # Retrieve the DataFrame payload.
        df_payload = control_message.payload()
        logger.debug("ImageCaptionTransformStage: Payload extracted with %d rows.", len(df_payload))

        # Remove the "caption" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "caption")
        logger.debug(
            "ImageCaptionTransformStage: Task configuration extracted: %s",
            pprint.pformat(sanitize_for_logging(task_config)),
        )

        # Call the caption extraction function.
        new_df = transform_image_create_vlm_caption_internal(
            df_payload, task_config=task_config, transform_config=self.validated_config
        )
        logger.info("Image caption extraction completed. New payload has %d rows.", len(new_df))

        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Optionally, annotate the control message with extraction trace info.
        # control_message.set_metadata("caption_extraction_trace", execution_trace_log)
        logger.info("ImageCaptionTransformStage.on_data: Updated control message and returning.")
        return control_message
