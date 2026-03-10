# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable, set_trace_timestamps_with_parent_context
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except
from typing import Optional

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook

logger = logging.getLogger(__name__)


@ray.remote
class InfographicExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts infographic data from image content.

    It expects an IngestControlMessage containing a DataFrame with image data. It then:
      1. Removes the "infographic_data_extract" task from the message.
      2. Calls the infographic extraction logic using a validated configuration.
      3. Updates the message payload with the extracted infographic DataFrame.
    """

    def __init__(self, config: InfographicExtractorSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=False, stage_name=stage_name)
        try:
            self.validated_config = config
            self._logger.info("InfographicExtractorStage configuration validated successfully.")
        except Exception as e:
            self._logger.exception(f"Error validating Infographic extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["infographic_data_extract"])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        # Extract DataFrame payload
        df_ledger = control_message.payload()

        # Remove the "infographic_data_extract" task from the message
        task_config = remove_task_by_type(control_message, "infographic_data_extract")

        execution_trace_log = {}
        new_df, extraction_info = extract_infographic_data_from_image_internal(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=execution_trace_log,
        )

        control_message.payload(new_df)
        control_message.set_metadata("infographic_extraction_info", extraction_info)

        do_trace_tagging = control_message.get_metadata("config::add_trace_tagging") is True
        if do_trace_tagging and execution_trace_log:
            parent_name = self.stage_name if self.stage_name else "infographic_extractor"
            set_trace_timestamps_with_parent_context(control_message, execution_trace_log, parent_name, logger)

        return control_message
