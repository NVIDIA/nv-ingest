# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
import logging
from typing import Optional
import pandas as pd
from pydantic import BaseModel
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.enums.common import (
    DocumentTypeEnum,
    ContentTypeEnum,
    AccessLevelEnum,
    TextTypeEnum,
    LanguageEnum,
)
from nv_ingest_api.internal.schemas.meta.metadata_schema import ContentHierarchySchema
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)


@ray.remote
class MetadataInjectionStage(RayActorStage):
    """
    A Ray actor stage that performs metadata injection on IngestControlMessages.

    This stage iterates over the rows of the DataFrame payload, checks if metadata
    injection is required, and if so, injects the appropriate metadata.
    """

    def __init__(self, config: BaseModel, stage_name: Optional[str] = None) -> None:
        # Call the base initializer to set attributes like self._running.
        super().__init__(config, stage_name=stage_name)
        # Additional initialization can be added here if necessary.
        self._logger.debug("MetadataInjectionStage initialized with config: %s", sanitize_for_logging(config))

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    def on_data(self, message: IngestControlMessage) -> IngestControlMessage:
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
        logger.debug("Starting metadata injection on DataFrame with %d rows", len(df))

        for _, row in df.iterrows():
            try:
                # Convert document type to content type using enums.
                content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
                # Check if metadata is missing or doesn't contain 'content'
                if (
                    "metadata" not in row
                    or not isinstance(row["metadata"], dict)
                    or "content" not in row["metadata"].keys()
                ):
                    update_required = True

                    # Initialize default structures based on MetaDataSchema
                    default_source_metadata = {
                        "source_id": row.get("source_id"),
                        "source_name": row.get("source_name"),
                        "source_type": row["document_type"],
                        "source_location": "",
                        "collection_id": "",
                        "date_created": datetime.now().isoformat(),
                        "last_modified": datetime.now().isoformat(),
                        "summary": "",
                        "partition_id": -1,
                        "access_level": AccessLevelEnum.UNKNOWN.value,
                    }

                    default_content_metadata = {
                        "type": content_type.name.lower(),
                        "page_number": -1,
                        "description": "",
                        "hierarchy": ContentHierarchySchema().model_dump(),
                        "subtype": "",
                        "start_time": -1,
                        "end_time": -1,
                    }

                    default_audio_metadata = None
                    if content_type == ContentTypeEnum.AUDIO:
                        default_audio_metadata = {
                            "audio_type": row["document_type"],
                            "audio_transcript": "",
                        }

                    default_image_metadata = None
                    if content_type == ContentTypeEnum.IMAGE:
                        default_image_metadata = {
                            "image_type": row["document_type"],
                            "structured_image_type": ContentTypeEnum.NONE.value,
                            "caption": "",
                            "text": "",
                            "image_location": (0, 0, 0, 0),
                            "image_location_max_dimensions": (0, 0),
                            "uploaded_image_url": "",
                            "width": 0,
                            "height": 0,
                        }

                    default_text_metadata = None
                    if content_type == ContentTypeEnum.TEXT:
                        default_text_metadata = {
                            "text_type": TextTypeEnum.DOCUMENT.value,
                            "summary": "",
                            "keywords": "",
                            "language": LanguageEnum.UNKNOWN.value,
                            "text_location": (0, 0, 0, 0),
                            "text_location_max_dimensions": (0, 0, 0, 0),
                        }

                    row["metadata"] = {
                        "content": row["content"],
                        "content_metadata": default_content_metadata,
                        "error_metadata": None,
                        "audio_metadata": default_audio_metadata,
                        "image_metadata": default_image_metadata,
                        "source_metadata": default_source_metadata,
                        "text_metadata": default_text_metadata,
                    }
                    logger.debug(
                        f"METADATA_INJECTOR_DEBUG: Rebuilt metadata for source_id='{row.get('source_id', 'N/A')}'. "
                        f"Metadata keys: {list(row['metadata'].keys())}."
                        f"'content' present: {'content' in row['metadata']}"
                    )
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

        return message
