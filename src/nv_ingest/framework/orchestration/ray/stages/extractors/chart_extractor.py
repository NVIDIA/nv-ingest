# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

import ray
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import set_trace_timestamps_with_parent_context
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


@ray.remote
class ChartExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts chart data from PDF content.

    It expects an IngestControlMessage containing a DataFrame payload with PDF documents.
    The stage removes the "chart_data_extract" task from the message, calls the internal
    extraction function using a validated ChartExtractorSchema, updates the message payload,
    and annotates the message metadata with extraction info.
    """

    def __init__(self, config: ChartExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            # logger.warning(
            #    "ChartExtractorStage validated config:\n%s", pprint.pformat(self.validated_config.model_dump())
            # )
        except Exception as e:
            logger.exception("Error validating chart extractor config")
            raise e

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["chart_data_extract"])
    def on_data(self, control_message: Any) -> Any:
        """
        Process the control message by extracting chart data.

        Parameters
        ----------
        control_message : IngestControlMessage
            The incoming message containing the PDF payload.

        Returns
        -------
        IngestControlMessage
            The updated message with the extracted chart data and extraction info in metadata.
        """
        logger.debug("ChartExtractorStage.on_data: Starting chart extraction.")
        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        logger.debug("ChartExtractorStage: Extracted payload with %d rows.", len(df_payload))

        # Remove the "chart_data_extract" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "chart_data_extract")
        logger.debug("ChartExtractorStage: Task config extracted: %s", sanitize_for_logging(task_config))

        # Perform chart data extraction.
        execution_trace_log = {}
        new_df, extraction_info = extract_chart_data_from_image_internal(
            df_extraction_ledger=df_payload,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=execution_trace_log,
        )
        logger.debug("ChartExtractorStage: Chart extraction completed. New payload has %d rows.", len(new_df))

        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Annotate the message with extraction info.
        control_message.set_metadata("chart_extraction_info", extraction_info)
        logger.debug("ChartExtractorStage: Metadata injection complete. Returning updated control message.")

        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            parent_name = self.stage_name if self.stage_name else "chart_extractor"
            set_trace_timestamps_with_parent_context(control_message, execution_trace_log, parent_name, logger)

        return control_message
