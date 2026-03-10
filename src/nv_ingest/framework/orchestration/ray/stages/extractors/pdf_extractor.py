# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any, Dict, Tuple, Optional
import ray

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema

from nv_ingest_api.internal.primitives.tracing.tagging import set_trace_timestamps_with_parent_context, traceable
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


def _inject_validated_config(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict,
    execution_trace_log: Optional[Any] = None,
    validated_config: Any = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Helper function that injects the validated_config into the configuration for PDF extraction
    and calls extract_primitives_from_pdf_internal.
    """
    return extract_primitives_from_pdf_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extractor_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


@ray.remote
class PDFExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts PDF primitives from a DataFrame payload.

    It expects an IngestControlMessage containing a DataFrame of PDF documents. It then:
      1. Removes the "extract" task from the message.
      2. Calls the PDF extraction logic (via _inject_validated_config) using a validated configuration.
      3. Updates the message payload with the extracted DataFrame.
      4. Optionally, stores additional extraction info in the message metadata.
    """

    def __init__(self, config: PDFExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            # Validate and store the PDF extractor configuration.
            self.validated_config = config
            logger.debug("PDFExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating PDF extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=[("extract", {"document_type": "pdf"})])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting PDF content.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with PDF documents.

        Returns
        -------
        IngestControlMessage
            The updated message with the extracted DataFrame and extraction info in metadata.
        """

        logger.debug("PDFExtractorStage.on_data: Starting PDF extraction process.")

        # Extract the DataFrame payload.
        df_extraction_ledger = control_message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_extraction_ledger))

        # Remove the "extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")
        logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

        # Perform PDF extraction.
        execution_trace_log = {}
        new_df, extraction_info = _inject_validated_config(
            df_extraction_ledger,
            task_config,
            execution_trace_log=execution_trace_log,
            validated_config=self.validated_config,
        )
        logger.debug("PDF extraction completed. Extracted %d rows.", len(new_df))

        # Update the message payload with the extracted DataFrame.
        control_message.payload(new_df)
        # Optionally, annotate the message with extraction info.
        control_message.set_metadata("pdf_extraction_info", extraction_info)
        logger.debug("PDF extraction metadata injected successfully.")

        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            # Use utility function to set trace timestamps with proper parent-child context
            parent_name = self.stage_name or "pdf_extractor"
            set_trace_timestamps_with_parent_context(control_message, execution_trace_log, parent_name, logger)

        return control_message
