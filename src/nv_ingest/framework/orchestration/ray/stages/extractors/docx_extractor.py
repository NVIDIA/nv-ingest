# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.extract.docx.docx_extractor import extract_primitives_from_docx_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


@ray.remote
class DocxExtractorStage(RayActorStage):
    """
    A Ray actor stage that extracts content from DOCX documents.

    It expects an IngestControlMessage containing a DataFrame with DOCX document data. It then:
      1. Removes the "docx-extract" task from the message.
      2. Calls the DOCX extraction logic (via extract_primitives_from_docx_internal) using a validated configuration.
      3. Updates the message payload with the extracted content DataFrame.
    """

    def __init__(self, config: DocxExtractorSchema) -> None:
        super().__init__(config, log_to_stdout=False)
        try:
            self.validated_config = config
            logger.info("DocxExtractorStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating DOCX Extractor config: {e}")
            raise

    @traceable("docx_extractor")
    @filter_by_task(required_tasks=[("extract", {"document_type": "docx"})])
    @nv_ingest_node_failure_try_except(annotation_id="docx_extractor", raise_on_failure=True)
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by extracting content from DOCX documents.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with DOCX document data.

        Returns
        -------
        IngestControlMessage
            The updated message with extracted DOCX content.
        """
        self._logger.debug("DocxExtractorStage.on_data: Starting DOCX extraction process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        self._logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "docx-extract" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "extract")
        self._logger.debug("Extracted task config: %s", task_config)

        # Perform DOCX content extraction.
        new_df, extraction_info = extract_primitives_from_docx_internal(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            extraction_config=self.validated_config,
            execution_trace_log=None,
        )

        # Update the message payload with the extracted DOCX content DataFrame.
        control_message.payload(new_df)
        control_message.set_metadata("docx_extraction_info", extraction_info)

        return control_message
