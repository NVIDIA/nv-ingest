# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import ray

from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except

logger = logging.getLogger(__name__)


@ray.remote
class InfographicExtractorStage(RayActorStage):
    def __init__(self, config: InfographicExtractorSchema) -> None:
        super().__init__(config)

        try:
            self.validated_config = config
            logger.info("ImageExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Image Extractor config: {e}")
            raise

    @traceable("infographic_extraction")
    @filter_by_task(required_tasks=["infographic_data_extract"])
    @nv_ingest_node_failure_try_except(annotation_id="infographic_extraction", raise_on_failure=False)
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
            for key, ts in execution_trace_log.items():
                control_message.set_timestamp(key, ts)

        return control_message
