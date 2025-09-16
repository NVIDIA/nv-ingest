# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook

logger = logging.getLogger(__name__)


@ray.remote
class TextSplitterStage(RayActorStage):
    """
    A Ray actor stage that splits documents into smaller parts based on specified criteria.

    This stage extracts the DataFrame payload from an IngestControlMessage, removes the "split"
    task (if present) to obtain the task configuration, and then calls the internal text splitting
    and tokenization logic. The updated DataFrame is then set back into the message.
    """

    def __init__(self, config: TextSplitterSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        # Store the validated configuration (assumed to be an instance of TextSplitterSchema)
        self.validated_config: TextSplitterSchema = config
        logger.info("TextSplitterStage initialized with config: %s", sanitize_for_logging(config))

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["split"])
    def on_data(self, message: Any) -> Any:
        """
        Process an incoming IngestControlMessage by splitting and tokenizing its text.

        Parameters
        ----------
        message : IngestControlMessage
            The incoming message containing the payload DataFrame.

        Returns
        -------
        IngestControlMessage
            The updated message with its payload transformed.
        """

        # Extract the DataFrame payload.
        df_payload = message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_payload))

        # Remove the "split" task to obtain task-specific configuration.
        task_config = remove_task_by_type(message, "split")
        logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

        # Transform the DataFrame (split text and tokenize).
        df_updated = transform_text_split_and_tokenize_internal(
            df_transform_ledger=df_payload,
            task_config=task_config,
            transform_config=self.validated_config,
            execution_trace_log=None,
        )
        logger.debug(
            "TextSplitterStage.on_data: Transformation complete. Updated payload has %d rows.", len(df_updated)
        )

        # Update the message payload.
        message.payload(df_updated)
        logger.debug("TextSplitterStage.on_data: Finished processing, returning updated message.")

        return message


def text_splitter_fn(control_message: IngestControlMessage, stage_config: TextSplitterSchema) -> IngestControlMessage:
    """
    Process an incoming IngestControlMessage by splitting and tokenizing its text.

    Parameters
    ----------
    control_message : IngestControlMessage
        The incoming message containing the payload DataFrame.

    stage_config : BaseModel
        The stage level configuration object

    Returns
    -------
    IngestControlMessage
        The updated message with its payload transformed.
    """

    # Extract the DataFrame payload.
    df_payload = control_message.payload()
    logger.debug("Extracted payload with %d rows.", len(df_payload))

    # Remove the "split" task to obtain task-specific configuration.
    task_config = remove_task_by_type(control_message, "split")
    logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

    # Transform the DataFrame (split text and tokenize).
    df_updated = transform_text_split_and_tokenize_internal(
        df_transform_ledger=df_payload,
        task_config=task_config,
        transform_config=stage_config,
        execution_trace_log=None,
    )
    logger.debug("TextSplitterStage.on_data: Transformation complete. Updated payload has %d rows.", len(df_updated))

    # Update the message payload.
    control_message.payload(df_updated)
    logger.debug("TextSplitterStage.on_data: Finished processing, returning updated message.")

    return control_message
