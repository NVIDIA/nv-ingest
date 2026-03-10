# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pprint
from typing import Optional
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type, IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook


@ray.remote
class TextEmbeddingTransformStage(RayActorStage):
    """
    A Ray actor stage that extracts text embeddings from a DataFrame payload.

    This stage uses the validated configuration (TextEmbeddingSchema) to process the DataFrame
    and generate text embeddings. The resulting DataFrame is set back on the message, and any
    trace or extraction metadata is added.
    """

    def __init__(self, config: TextEmbeddingSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            self._logger.info("TextEmbeddingTransformStage configuration validated successfully.")
        except Exception as e:
            self._logger.exception(f"Error validating text embedding config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["embed"])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by generating text embeddings.

        Parameters
        ----------
        control_message : IngestControlMessage
            The incoming message containing the DataFrame payload.

        Returns
        -------
        IngestControlMessage
            The updated message with text embeddings and trace info added.
        """
        # Get the DataFrame payload.
        df_payload = control_message.payload()
        self._logger.debug("TextEmbeddingTransformStage: Extracted payload with %d rows.", len(df_payload))

        # Remove the "embed" task to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "embed")
        self._logger.debug(
            "TextEmbeddingTransformStage: Task configuration extracted: %s",
            pprint.pformat(sanitize_for_logging(task_config)),
        )

        # Call the text embedding extraction function.
        new_df, execution_trace_log = transform_create_text_embeddings_internal(
            df_payload, task_config=task_config, transform_config=self.validated_config
        )

        # Update the control message payload.
        control_message.payload(new_df)
        # Annotate the message metadata with trace info.
        control_message.set_metadata("text_embedding_trace", execution_trace_log)
        return control_message
