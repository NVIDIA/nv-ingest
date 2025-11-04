# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.image_extractor import extract_primitives_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook

logger = logging.getLogger(__name__)


@ray.remote
class ImageExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts primitives from image content.

    It expects an IngestControlMessage containing a DataFrame with image data. It then:
      1. Removes the "extract" task from the message.
      2. Calls the image extraction logic (via extract_primitives_from_image_internal) using a validated configuration.
      3. Updates the message payload with the extracted primitives DataFrame.
    """

    def __init__(self, config: ImageExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=False, stage_name=stage_name)
        try:
            self.validated_config = config
            self._logger.info("ImageExtractorStage configuration validated successfully.")
        except Exception as e:
            self._logger.exception(f"Error validating Image Extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=[("extract", {"document_type": "regex:^(png|jpeg|jpg|tiff|bmp)$"})])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by extracting primitives from images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image data.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted image primitives.
        """
        logger.info("ImageExtractorStage.on_data: Starting image extraction process.")
        try:
            # Extract the DataFrame payload.
            df_ledger = control_message.payload()
            logger.debug("Extracted payload with %d rows.", len(df_ledger))

            # Remove the "extract" task from the message to obtain task-specific configuration.
            task_config = remove_task_by_type(control_message, "extract")
            logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

            # Perform image primitives extraction.
            new_df, extraction_info = extract_primitives_from_image_internal(
                df_extraction_ledger=df_ledger,
                task_config=task_config,
                extraction_config=self.validated_config,
                execution_trace_log=None,
            )
            logger.info("Image extraction completed. Resulting DataFrame has %d rows.", len(new_df))

            # Update the message payload with the extracted primitives DataFrame.
            control_message.payload(new_df)
            control_message.set_metadata("image_extraction_info", extraction_info)

            return control_message
        except Exception as e:
            logger.exception(f"ImageExtractorStage failed processing control message: {e}")
            raise
