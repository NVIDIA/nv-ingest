# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
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
    Helper function that injects the validated_config into the configuration for Chart extraction
    and calls extract_chart_data_from_image_internal.
    """
    return extract_chart_data_from_image_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


class PythonChartExtractorStage(PythonStage):
    """
    A Python stage that extracts chart data from images.

    It expects an IngestControlMessage containing a DataFrame with image data containing charts. It then:
      1. Removes the "extract_charts" task from the message.
      2. Calls the chart extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted chart data DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: ChartExtractorSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("PythonChartExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Chart Extractor config: {e}")
            raise

    @traceable("chart_extractor")
    @filter_by_task(required_tasks=[("extract_charts", {})])
    @nv_ingest_node_failure_try_except(annotation_id="chart_extractor", raise_on_failure=False)
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting chart data from images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image data containing charts.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted chart data.
        """
        self._logger.debug("PythonChartExtractorStage.on_data: Starting chart extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "extract_charts" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract_charts")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform chart data extraction.
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            execution_trace_log=None,
            validated_config=self.validated_config,
        )

        # Update the message payload with the extracted chart data DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("chart_extraction_info", extraction_info)

        # Update statistics
        self.stats["processed"] += 1

        # Add trace tagging if enabled
        if hasattr(control_message, "get_metadata") and control_message.get_metadata("trace_enabled", False):
            control_message.set_metadata("chart_extractor_processed", True)

        self._logger.debug("PythonChartExtractorStage.on_data: Chart extraction completed.")
        return control_message
