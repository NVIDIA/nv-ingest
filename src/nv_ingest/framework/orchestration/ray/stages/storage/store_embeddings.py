# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.store.embed_text_upload import store_text_embeddings_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook

logger = logging.getLogger(__name__)


@ray.remote
class EmbeddingStorageStage(RayActorStage):
    """
    A Ray actor stage that stores text embeddings in MinIO.

    It expects an IngestControlMessage containing a DataFrame with embedding data. It then:
      1. Removes the "store_embedding" task from the message.
      2. Calls the embedding storage logic (via store_text_embeddings_internal) using a validated configuration.
      3. Updates the message payload with the stored embeddings DataFrame.
    """

    def __init__(self, config: EmbeddingStorageSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        try:
            self.validated_config = config
            logger.info("EmbeddingStorageStage configuration validated successfully.")
        except Exception as e:
            logger.exception(f"Error validating Embedding Storage config: {e}")
            raise

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    @filter_by_task(required_tasks=["store_embedding"])
    def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
        """
        Process the control message by storing embeddings.

        Parameters
        ----------
        control_message : IngestControlMessage
            The message containing a DataFrame payload with embedding data.

        Returns
        -------
        IngestControlMessage
            The updated message with embeddings stored in MinIO.
        """
        logger.info("EmbeddingStorageStage.on_data: Starting embedding storage process.")

        # Extract the DataFrame payload.
        df_ledger = control_message.payload()
        logger.debug("Extracted payload with %d rows.", len(df_ledger))

        # Remove the "store_embedding" task from the message to obtain task-specific configuration.
        task_config = remove_task_by_type(control_message, "store_embedding")
        logger.debug("Extracted task config: %s", sanitize_for_logging(task_config))

        # Perform embedding storage.
        new_df = store_text_embeddings_internal(
            df_store_ledger=df_ledger,
            task_config=task_config,
            store_config=self.validated_config,
            execution_trace_log=None,
        )
        logger.info("Embedding storage completed. Resulting DataFrame has %d rows.", len(new_df))

        # Update the message payload with the stored embeddings DataFrame.
        control_message.payload(new_df)

        return control_message
