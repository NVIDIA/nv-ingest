# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
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
    Helper function that injects the validated_config into the configuration for Infographic extraction
    and calls extract_infographic_data_from_image_internal.
    """
    return extract_infographic_data_from_image_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


class PythonInfographicExtractorStage(PythonStage):
    """
    A Python stage that extracts infographic data from images.

    It expects an IngestControlMessage containing a DataFrame with image data containing infographics. It then:
      1. Removes the "infographic_data_extract" task from the message.
      2. Calls the infographic extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted infographic data DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: InfographicExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("PythonInfographicExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Infographic Extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except(annotation_id="infographic_extraction", raise_on_failure=False)
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["infographic_data_extract"])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting infographic data from images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image data containing infographics.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted infographic data.
        """
        self._logger.debug("PythonInfographicExtractorStage.on_data: Starting infographic extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "infographic_data_extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "infographic_data_extract")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform infographic data extraction.
        execution_trace_log = {}
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            execution_trace_log=execution_trace_log,
            validated_config=self.validated_config,
        )

        # Update the message payload with the extracted infographic data DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("infographic_extraction_info", extraction_info)

        # Update statistics
        self.stats["processed"] += 1

        # Add trace tagging if enabled
        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            for key, ts in execution_trace_log.items():
                control_message.set_timestamp(key, ts)

        self._logger.debug("PythonInfographicExtractorStage.on_data: Infographic extraction completed.")
        return control_message
