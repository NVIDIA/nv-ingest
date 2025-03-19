# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pprint
from typing import Any

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_context_manager,
    unified_exception_handler,
)

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

    def __init__(self, config: ChartExtractorSchema, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        try:
            self.validated_config = config
            logger.warning(pprint.pformat(self.validated_config.model_dump()))
        except Exception as e:
            logger.exception("Error validating chart extractor config")
            raise e

    @filter_by_task(required_tasks=["chart_data_extract"])
    @nv_ingest_node_failure_context_manager(annotation_id="chart_extraction", raise_on_failure=False)
    @unified_exception_handler
    async def on_data(self, control_message: Any) -> Any:
        # Extract the DataFrame payload.
        df_payload = control_message.payload()
        # Remove the "chart_data_extract" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "chart_data_extract")
        # Perform chart data extraction.
        new_df, extraction_info = extract_chart_data_from_image_internal(
            df_extraction_ledger=df_payload,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=None,
        )
        # Update the control message with the new DataFrame.
        control_message.payload(new_df)
        # Annotate the message with extraction info.
        control_message.set_metadata("chart_extraction_info", extraction_info)
        return control_message
