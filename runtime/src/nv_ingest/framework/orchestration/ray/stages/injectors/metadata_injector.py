# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd
from typing import Any
from pydantic import BaseModel
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.internal.enums.common import DocumentTypeEnum, ContentTypeEnum
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@ray.remote
class MetadataInjectionStage(RayActorStage):
    """
    A Ray actor stage that performs metadata injection on IngestControlMessages.

    This stage iterates over the rows of the DataFrame payload, checks if metadata
    injection is required, and if so, injects the appropriate metadata.
    """

    def __init__(self, config: BaseModel) -> None:
        # Call the base initializer to set attributes like self._running.
        super().__init__(config)
        # Additional initialization can be added here if necessary.
        logger.info("MetadataInjectionStage initialized with config: %s", config)

    @traceable("metadata_injector")
    @nv_ingest_node_failure_try_except(annotation_id="metadata_injector", raise_on_failure=False)
    def on_data(self, message: Any) -> Any:
        """
        Process an incoming IngestControlMessage by injecting metadata into its DataFrame payload.

        Parameters
        ----------
        message : IngestControlMessage
            The incoming message containing the payload DataFrame.

        Returns
        -------
        IngestControlMessage
            The message with updated metadata if injection was required.
        """
        df = message.payload()
        update_required = False
        rows = []
        logger.info("Starting metadata injection on DataFrame with %d rows", len(df))

        for _, row in df.iterrows():
            try:
                # Convert document type to content type using enums.
                content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
                # Check if metadata is missing or doesn't contain 'content'
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
            logger.info("Metadata injection updated payload with %d rows", len(docs))
        else:
            logger.info("No metadata update was necessary during metadata injection")

        return message
