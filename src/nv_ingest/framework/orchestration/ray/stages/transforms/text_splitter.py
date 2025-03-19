import logging
from typing import Any
from pydantic import BaseModel
import ray

# Assume these imports come from your project:
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager

logger = logging.getLogger(__name__)


@ray.remote
class TextSplitterStage(RayActorStage):
    """
    A Ray actor stage that splits documents into smaller parts based on specified criteria.

    This stage extracts the DataFrame payload from an IngestControlMessage, removes the "split"
    task (if present) to obtain the task configuration, and then calls the internal text splitting
    and tokenization logic. The updated DataFrame is then set back into the message.
    """

    def __init__(self, config: BaseModel, progress_engine_count: int) -> None:
        # Initialize the base class so that attributes like self.running are properly set.
        super().__init__(config, progress_engine_count)
        # Store the validated configuration (assumed to be an instance of TextSplitterSchema)
        self.validated_config: TextSplitterSchema = config

    @filter_by_task(["split"])
    @nv_ingest_node_failure_context_manager(annotation_id="text_splitter", raise_on_failure=False)
    async def on_data(self, message: Any) -> Any:
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
        try:
            # Extract the DataFrame payload.
            df_payload = message.payload()
            # Remove the "split" task to obtain task-specific configuration.
            task_config = remove_task_by_type(message, "split")

            # Transform the DataFrame (split text and tokenize).
            df_updated = transform_text_split_and_tokenize_internal(
                df_transform_ledger=df_payload,
                task_config=task_config,
                transform_config=self.validated_config,
                execution_trace_log=None,
            )

            # Update the message payload.
            message.payload(df_updated)
            return message
        except Exception as e:
            logger.exception("TextSplitterStage failed to process IngestControlMessage")
            raise type(e)(f"TextSplitterStage: {str(e)}") from e
