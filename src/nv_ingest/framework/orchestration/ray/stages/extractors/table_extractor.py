# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
import ray

# These imports are assumed from your project.
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.table_extractor import extract_table_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

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

    def __init__(self, config: TableExtractorSchema) -> None:
        super().__init__(config)
        try:
            self.validated_config = config
            logger.info("TableExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception("Error validating table extractor config")
            raise e

    @traceable("table_extraction")
    @filter_by_task(required_tasks=["table_data_extract"])
    @nv_ingest_node_failure_try_except(annotation_id="table_extraction", raise_on_failure=False)
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
        logger.info("TableExtractorStage.on_data: Starting table extraction.")
        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_payload))

        # Remove the "table_data_extract" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "table_data_extract")
        logger.debug("Extracted task configuration: %s", task_config)

        # Perform table data extraction.
        execution_trace_log = {}
        new_df, extraction_info = extract_table_data_from_image_internal(
            df_extraction_ledger=df_payload,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=execution_trace_log,
        )
        logger.info("Table extraction completed. Extracted %d rows.", len(new_df))

        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Annotate the message with extraction info.
        control_message.set_metadata("table_extraction_info", extraction_info)
        logger.info("Table extraction metadata injected successfully.")

        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            for key, ts in execution_trace_log.items():
                control_message.set_timestamp(key, ts)

        return control_message
