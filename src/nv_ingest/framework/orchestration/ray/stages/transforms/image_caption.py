# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_context_manager,
    unified_exception_handler,
)

logger = logging.getLogger(__name__)


@ray.remote
class ImageCaptionTransformStage(RayActorStage):
    """
    A Ray actor stage that extracts image captions from a DataFrame payload.

    This stage validates its configuration (using ImageCaptionExtractionSchema), then processes the DataFrame
    via transform_image_create_vlm_caption_internal. The updated DataFrame and any extraction trace info
    are stored in the control message.
    """

    def __init__(self, config: ImageCaptionExtractionSchema, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        try:
            self.validated_config = config
        except Exception as e:
            logger.exception("Error validating caption extraction config")
            raise e

    @filter_by_task(required_tasks=["caption"])
    @nv_ingest_node_failure_context_manager(annotation_id="image_captioning", raise_on_failure=False)
    @unified_exception_handler
    async def on_data(self, control_message: Any) -> Any:
        # Retrieve the DataFrame payload.
        df_payload = control_message.payload()
        # Call the caption extraction function.
        task_config = remove_task_by_type(control_message, "caption")
        new_df = transform_image_create_vlm_caption_internal(
            df_payload, task_config=task_config, transform_config=self.validated_config
        )
        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Annotate the control message with extraction trace info.
        # control_message.set_metadata("caption_extraction_trace", execution_trace_log)
        return control_message
