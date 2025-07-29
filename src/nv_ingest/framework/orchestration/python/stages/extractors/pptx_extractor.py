# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional

from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.pptx.pptx_extractor import extract_primitives_from_pptx_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
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
    Helper function that injects the validated_config into the configuration for PPTX extraction
    and calls extract_primitives_from_pptx_internal.
    """
    return extract_primitives_from_pptx_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


class PythonPPTXExtractorStage(PythonStage):
    """
    A Python stage that extracts content from PPTX documents.

    It expects an IngestControlMessage containing a DataFrame with PPTX document data. It then:
      1. Removes the "extract" task from the message.
      2. Calls the PPTX extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted content DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: PPTXExtractorSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("PythonPPTXExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating PPTX Extractor config: {e}")
            raise

    @traceable("pptx_extractor")
    @filter_by_task(required_tasks=[("extract", {"document_type": "pptx"})])
    @nv_ingest_node_failure_try_except(annotation_id="pptx_extractor", raise_on_failure=False)
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting content from PPTX documents.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with PPTX document data.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted PPTX content.
        """
        self._logger.debug("PythonPPTXExtractorStage.on_data: Starting PPTX extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform PPTX content extraction.
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            execution_trace_log=None,
            validated_config=self.validated_config,
        )

        # Update the message payload with the extracted PPTX content DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("pptx_extraction_info", extraction_info)

        # Update statistics
        self.stats["processed"] += 1

        # Add trace tagging if enabled
        if hasattr(control_message, "get_metadata") and control_message.get_metadata("trace_enabled", False):
            control_message.set_metadata("pptx_extractor_processed", True)

        self._logger.debug("PythonPPTXExtractorStage.on_data: PPTX extraction completed.")
        return control_message
