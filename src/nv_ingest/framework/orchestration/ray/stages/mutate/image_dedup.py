# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.mutate.deduplicate import deduplicate_images_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


@ray.remote
class ImageDedupStage(RayActorStage):
    """
    A Ray actor stage that deduplicates images within a DataFrame payload.

    It expects an IngestControlMessage containing a DataFrame with image documents. It then:
      1. Removes the "dedup" task from the message.
      2. Calls the image deduplication logic (via deduplicate_images_internal) using a validated configuration.
      3. Updates the message payload with the deduplicated DataFrame.
    """

    def __init__(self, config: ImageDedupSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.debug("ImageDedupStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Image Deduplication config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["dedup"])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by deduplicating images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image documents.

        Returns
        -------
        IngestControlMessage
            The updated message with deduplicated images in the payload.
        """
        logger.debug("ImageDedupStage.on_data: Starting image deduplication process.")
        try:
            # Extract the DataFrame payload.
            df_ledger = control_message.payload()
            logger.debug("Extracted payload with %d rows.", len(df_ledger))

            # Remove the "dedup" task from the message to obtain task-specific configuration.
            task_config = remove_task_by_type(control_message, "dedup")
            logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

            # Perform image deduplication.
            new_df = deduplicate_images_internal(
                df_ledger=df_ledger,
                task_config=task_config,
                mutate_config=self.validated_config,
                execution_trace_log=None,
            )
            logger.debug("Image deduplication completed. Resulting DataFrame has %d rows.", len(new_df))

            # Update the message payload with the deduplicated DataFrame.
            control_message.payload(new_df)

            return control_message
        except Exception as e:
            logger.exception(f"ImageDedupStage failed processing control message: {e}")
            raise
