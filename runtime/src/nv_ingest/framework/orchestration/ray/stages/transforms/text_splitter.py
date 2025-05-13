import logging
from typing import Any
import ray

# Assume these imports come from your project:
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


@ray.remote
class TextSplitterStage(RayActorStage):
    """
    A Ray actor stage that splits documents into smaller parts based on specified criteria.

    This stage extracts the DataFrame payload from an IngestControlMessage, removes the "split"
    task (if present) to obtain the task configuration, and then calls the internal text splitting
    and tokenization logic. The updated DataFrame is then set back into the message.
    """

    def __init__(self, config: TextSplitterSchema) -> None:
        super().__init__(config)
        # Store the validated configuration (assumed to be an instance of TextSplitterSchema)
        self.validated_config: TextSplitterSchema = config
        logger.info("TextSplitterStage initialized with config: %s", config)

    @traceable("text_splitter")
    @filter_by_task(["split"])
    @nv_ingest_node_failure_try_except(annotation_id="text_splitter", raise_on_failure=False)
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
        logger.debug("Extracted task config: %s", task_config)

        # Transform the DataFrame (split text and tokenize).
        df_updated = transform_text_split_and_tokenize_internal(
            df_transform_ledger=df_payload,
            task_config=task_config,
            transform_config=self.validated_config,
            execution_trace_log=None,
        )
        logger.info("TextSplitterStage.on_data: Transformation complete. Updated payload has %d rows.", len(df_updated))

        # Update the message payload.
        message.payload(df_updated)
        logger.info("TextSplitterStage.on_data: Finished processing, returning updated message.")

        return message
