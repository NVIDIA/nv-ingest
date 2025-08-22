# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.extract.image.table_extractor import extract_table_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
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
    Helper function that injects the validated_config into the configuration for Table extraction
    and calls extract_table_data_from_image_internal.
    """
    return extract_table_data_from_image_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


class PythonTableExtractorStage(PythonStage):
    """
    A Python stage that extracts table data from images.

    It expects an IngestControlMessage containing a DataFrame with image data containing tables. It then:
      1. Removes the "extract_tables" task from the message.
      2. Calls the table extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted table data DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: TableExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("PythonTableExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Table Extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except(annotation_id="table_extractor", raise_on_failure=False)
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=[("extract_tables", {})])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting table data from images.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with image data containing tables.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted table data.
        """
        self._logger.debug("PythonTableExtractorStage.on_data: Starting table extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "extract_tables" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract_tables")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform table data extraction.
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            execution_trace_log=None,
            validated_config=self.validated_config,
        )

        # Update the message payload with the extracted table data DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("table_extraction_info", extraction_info)

        # Update statistics
        self.stats["processed"] += 1

        # Add trace tagging if enabled
        if hasattr(control_message, "get_metadata") and control_message.get_metadata("trace_enabled", False):
            control_message.set_metadata("table_extractor_processed", True)

        self._logger.debug("PythonTableExtractorStage.on_data: Table extraction completed.")
        return control_message
