import asyncio
import json
import logging
from typing import Any, Dict

import ray
import pandas as pd

from nv_ingest_api.internal.enums.common import DocumentTypeEnum, ContentTypeEnum

# Import your Morpheus primitives.
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type

logger = logging.getLogger(__name__)


def on_data(message: IngestControlMessage) -> IngestControlMessage:
    """
    Process a job by injecting metadata into its payload.

    Parameters
    ----------
    message : IngestControlMessage
        The incoming control message.

    Returns
    -------
    IngestControlMessage
        The control message after metadata injection.
    """
    try:
        df = message.payload()
        logger.warning(f"[MDI PAYLOAD IN]: {df}\n")
        update_required = False
        rows = []
        logger.debug("Starting metadata injection on DataFrame with %d rows", len(df))

        for _, row in df.iterrows():
            try:
                content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
                if "metadata" not in row or not isinstance(row["metadata"], dict) or "content" not in row["metadata"]:
                    update_required = True
                    row["metadata"] = {
                        "content": row.get("content"),
                        "content_metadata": {
                            "type": content_type.name.lower(),
                        },
                        "error_metadata": None,
                        "audio_metadata": (
                            None if content_type != ContentTypeEnum.AUDIO else {"audio_type": row["document_type"]}
                        ),
                        "image_metadata": (
                            None if content_type != ContentTypeEnum.IMAGE else {"image_type": row["document_type"]}
                        ),
                        "source_metadata": {
                            "source_id": row.get("source_id"),
                            "source_name": row.get("source_name"),
                            "source_type": row["document_type"],
                        },
                        "text_metadata": (None if content_type != ContentTypeEnum.TEXT else {"text_type": "document"}),
                    }
            except Exception as inner_e:
                logger.exception("Failed to process row during metadata injection")
                raise inner_e
            rows.append(row)

        if update_required:
            docs = pd.DataFrame(rows)
            message.payload(docs)
            logger.debug("Metadata injection updated payload with %d rows", len(docs))
        else:
            logger.debug("No metadata update was necessary during metadata injection")

        logger.warning(f"[MDI PAYLOAD OUT]: {message.payload()}\n")
        logger.warning(f"[MDI PAYLOAD OUT]: {json.dumps(message.payload().iloc[0]['metadata'], indent=2)}\n")
        return message

    except Exception as e:
        new_message = f"on_data: Failed to process IngestControlMessage. Original error: {str(e)}"
        logger.exception(new_message)
        raise type(e)(new_message) from e


@ray.remote
class MetadataInjectionStage:
    """
    A Ray actor stage for injecting metadata into control messages.

    This stage wraps the synchronous on_data function into an asynchronous process() method
    so that it can be integrated into a streaming pipeline.

    Attributes
    ----------
    config : Dict[str, Any]
        Configuration parameters for the stage.
    downstream_queue : Any, optional
        The Ray actor handle representing the downstream queue.
    """

    def __init__(self, **config: Any) -> None:
        """
        Initialize the MetadataInjectionStage.

        Parameters
        ----------
        **config : dict
            Additional configuration parameters.
        """
        self.config: Dict[str, Any] = config
        self.downstream_queue: Any = None

    async def process(self, message: IngestControlMessage) -> IngestControlMessage:
        """
        Asynchronously process a control message by injecting metadata.

        Parameters
        ----------
        message : IngestControlMessage
            The incoming control message.

        Returns
        -------
        IngestControlMessage
            The control message after metadata injection.
        """
        # Run the blocking on_data in a separate thread.
        result = await asyncio.to_thread(on_data, message)
        # Optionally forward the result to a downstream queue if set.
        if self.downstream_queue:
            await self.downstream_queue.put.remote(result)
        return result

    def set_output_queue(self, queue_handle: Any) -> bool:
        """
        Set the downstream queue for this stage.

        Parameters
        ----------
        queue_handle : Any
            The Ray actor handle representing the downstream queue.

        Returns
        -------
        bool
            True if the downstream queue was set successfully.
        """
        self.downstream_queue = queue_handle
        return True
