# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.extract.image.table_extractor import extract_table_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable, set_trace_timestamps_with_parent_context
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


@ray.remote
class TableExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts table data from PDF content.

    It expects an IngestControlMessage containing a DataFrame payload with PDF documents.
    The stage removes the "table_data_extract" task from the message, calls the internal
    extraction function using a validated TableExtractorSchema, updates the message payload,
    and annotates the message metadata with extraction info.
    """

    def __init__(self, config: TableExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.debug("TableExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception("Error validating table extractor config")
            raise e

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["table_data_extract"])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting table data from the PDF payload.

        Parameters
        ----------
        control_message : IngestControlMessage
            The incoming message containing the PDF payload.

        Returns
        -------
        IngestControlMessage
            The updated message with the extracted table data and extraction info in metadata.
        """
        logger.debug("TableExtractorStage.on_data: Starting table extraction.")
        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_payload))

        # Remove the "table_data_extract" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "table_data_extract")
        logger.debug("Extracted task configuration: %s", sanitize_for_logging(task_config))

        # Perform table data extraction.
        execution_trace_log = {}
        new_df, extraction_info = extract_table_data_from_image_internal(
            df_extraction_ledger=df_payload,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=execution_trace_log,
        )
        logger.debug("Table extraction completed. Extracted %d rows.", len(new_df))

        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Annotate the message with extraction info.
        control_message.set_metadata("table_extraction_info", extraction_info)
        logger.debug("Table extraction metadata injected successfully.")

        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            parent_name = self.stage_name if self.stage_name else "table_extractor"
            set_trace_timestamps_with_parent_context(control_message, execution_trace_log, parent_name, logger)

        return control_message
