# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import ray
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.pptx.pptx_extractor import extract_primitives_from_pptx_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except

logger = logging.getLogger(__name__)


@ray.remote
class PPTXExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts content from PPTX documents.

    It expects an IngestControlMessage containing a DataFrame with PPTX document data. It then:
      1. Removes the "pptx-extract" task from the message.
      2. Calls the PPTX extraction logic (via extract_primitives_from_pptx_internal) using a validated configuration.
      3. Updates the message payload with the extracted content DataFrame.
    """

    def __init__(self, config: PPTXExtractorSchema, stage_name: Optional[str] = None) -> None:
        """
        Initializes the PptxExtractorStage.

        Parameters
        ----------
        config : PPTXExtractorSchema
            The validated configuration object for PPTX extraction.
        stage_name : Optional[str]
            Name of the stage from YAML pipeline configuration.
        """
        super().__init__(config, stage_name=stage_name)
        try:
            # The config passed in should already be validated, but storing it.
            self.validated_config = config
            logger.info("PptxExtractorStage configuration validated successfully.")
        except Exception as e:
            # If RayActorStage.__init__ or config access raises an issue.
            logger.exception(f"Error initializing or validating PPTX Extractor config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=[("extract", {"document_type": "pptx"})])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
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

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()

        # Remove the "pptx-extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")

        new_df, extraction_info = extract_primitives_from_pptx_internal(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=None,  # Assuming None is appropriate here as in DOCX example
        )

        # Update the message payload with the extracted PPTX content DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("pptx_extraction_info", extraction_info)

        return control_message
