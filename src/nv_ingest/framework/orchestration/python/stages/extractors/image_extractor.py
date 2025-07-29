# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.image_extractor import extract_primitives_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


def _inject_validated_config(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict,
    execution_trace_log: Optional[Any] = None,
    validated_config: Any = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Helper function that injects the validated_config into the configuration for Image extraction
    and calls extract_primitives_from_image_internal.
    """
    return extract_primitives_from_image_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


class PythonImageExtractorStage(PythonStage):
    """
    A Python stage that extracts primitives from image content.

    It expects an IngestControlMessage containing a DataFrame with image data. It then:
      1. Removes the "extract" task from the message.
      2. Calls the image extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted primitives DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: ImageExtractorSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("PythonImageExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Image Extractor config: {e}")
            raise

    @traceable("image_extraction")
    @filter_by_task(required_tasks=[("extract", {"document_type": "regex:^(png|jpeg|jpg|tiff|bmp)$"})])
    @nv_ingest_node_failure_try_except(annotation_id="image_extractor", raise_on_failure=False)
    def on_data(self, control_message: Any) -> Any:
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
        self._logger.info("PythonImageExtractorStage.on_data: Starting image extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform image primitives extraction.
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            execution_trace_log=None,
            validated_config=self.validated_config,
        )
        self._logger.info("Image extraction completed. Resulting DataFrame has %d rows.", len(new_df))

        # Update the message payload with the extracted primitives DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("image_extraction_info", extraction_info)

        # Update statistics
        self.stats["processed"] += 1

        # Add trace tagging if enabled
        if hasattr(control_message, "get_metadata") and control_message.get_metadata("trace_enabled", False):
            control_message.set_metadata("image_extractor_processed", True)

        self._logger.debug("PythonImageExtractorStage.on_data: Image extraction completed.")
        return control_message
