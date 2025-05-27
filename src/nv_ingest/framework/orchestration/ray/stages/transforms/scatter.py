# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid

import ray
from typing import Callable, List, Any, Optional, Dict

from pydantic import BaseModel, Field

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.orchestration.ray.util.pipeline.scatterers.pdf_fragmenter import create_pdf_fragmenter
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except

logger = logging.getLogger(__name__)


class LambdaScatterSchema(BaseModel):
    """Configuration schema for LambdaScatterStage"""

    # Add any configuration parameters you need
    max_output_messages: Optional[int] = None  # Optional limit on output messages
    task_config_passthrough: bool = True  # Whether to pass task config to callable
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


@ray.remote
class LambdaScatterStage(RayActorStage):
    """
    A Ray actor stage that applies a user-defined callable to scatter/transform messages.

    This stage takes an IngestControlMessage and applies a callable function that can produce
    1 to N output IngestControlMessages. This enables flexible message transformation and
    routing patterns within the pipeline.
    """

    def __init__(
        self,
        config: LambdaScatterSchema,
        transform_callable: Callable[[IngestControlMessage], List[IngestControlMessage]],
    ) -> None:
        """
        Initialize the LambdaScatterStage.

        Parameters
        ----------
        config : LambdaScatterSchema
            Configuration schema for the stage
        transform_callable : Callable[[IngestControlMessage], List[IngestControlMessage]]
            A callable that takes an IngestControlMessage and returns a list of 1 to N
            IngestControlMessages
        """
        super().__init__(config)
        # Store the validated configuration
        self.validated_config: LambdaScatterSchema = config
        self._transform_callable = transform_callable

        logger.info("LambdaScatterStage initialized with config: %s", config)

    @traceable("lambda_scatter")
    @nv_ingest_node_failure_try_except(annotation_id="lambda_scatter", raise_on_failure=False)
    def on_data(self, message: Any) -> List[Any]:
        """
        Process an incoming IngestControlMessage by applying the transform callable.

        Parameters
        ----------
        message : IngestControlMessage
            The incoming message to be transformed.

        Returns
        -------
        List[IngestControlMessage]
            A list of 1 to N transformed messages, each annotated with fragment metadata.
        """
        logger.debug("LambdaScatterStage.on_data: Processing incoming message")

        try:
            output_messages = self._transform_callable(message)

            if not isinstance(output_messages, list):
                raise ValueError(f"Transform callable must return a list, got {type(output_messages)}")

            if len(output_messages) == 0:
                raise ValueError("Transform callable must return at least 1 message")

            for idx, msg in enumerate(output_messages):
                if not isinstance(msg, IngestControlMessage):
                    raise ValueError(f"Output message at index {idx} is not an IngestControlMessage")

            if len(output_messages) > 1:
                fragment_id = str(uuid.uuid4())
                for idx, msg in enumerate(output_messages):
                    msg.set_metadata("fragment_id", fragment_id)
                    msg.set_metadata("fragment_count", len(output_messages))
                    msg.set_metadata("fragment_index", idx)

            logger.info("LambdaScatterStage.on_data: Transform produced %d output messages", len(output_messages))

        except Exception as e:
            logger.error("LambdaScatterStage.on_data: Transform callable failed: %s", str(e))
            raise

        return output_messages


class PDFScatterSchema(BaseModel):
    """Configuration schema for PdfScatterStage"""

    pages_per_fragment: int = Field(default=100, description="Number of pages per fragment")
    max_fragments: int = Field(default=1000, description="Maximum number of fragments to create")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")


@ray.remote
class PDFScatterStage(RayActorStage):
    """
    A Ray actor stage that fragments PDF documents into smaller chunks.

    This stage takes an IngestControlMessage containing a PDF and splits it into
    multiple messages, each containing a fragment of the original PDF with a
    configurable number of pages per fragment.
    """

    def __init__(self, config: PDFScatterSchema) -> None:
        """
        Initialize the PdfScatterStage.

        Parameters
        ----------
        config : PDFScatterSchema
            Configuration for PDF fragmentation
        """
        super().__init__(config)
        self.validated_config: PDFScatterSchema = config

        # Create the PDF fragmenter with the configured parameters
        self._pdf_fragmenter = create_pdf_fragmenter(
            pages_per_fragment=config.pages_per_fragment,
        )

        logger.info(
            "PdfScatterStage initialized with pages_per_fragment=%d, overlap=%s, overlap_pages=%d",
            config.pages_per_fragment,
        )

    # TODO(Filter by docutype for slight optimization)
    @traceable("pdf_scatter")
    @nv_ingest_node_failure_try_except(annotation_id="pdf_scatter", raise_on_failure=False)
    def on_data(self, message: Any) -> List[Any]:
        """
        Process an incoming IngestControlMessage by fragmenting any PDFs.

        Parameters
        ----------
        message : IngestControlMessage
            The incoming message potentially containing a PDF

        Returns
        -------
        List[IngestControlMessage]
            A list of messages, each containing a PDF fragment.
            Non-PDF messages are returned unchanged as a single-item list.
        """
        logger.debug("PdfScatterStage.on_data: Processing incoming message")

        # Apply PDF fragmentation
        output_messages = self._pdf_fragmenter(message)

        # Validate we don't exceed max fragments
        if len(output_messages) > self.validated_config.max_fragments:
            logger.warning(
                "PDF fragmentation produced %d fragments, exceeding max_fragments=%d. Truncating.",
                len(output_messages),
                self.validated_config.max_fragments,
            )
            output_messages = output_messages[: self.validated_config.max_fragments]

        # Annotate fragments if we have more than one
        if len(output_messages) > 1:
            fragment_id = str(uuid.uuid4())
            for idx, msg in enumerate(output_messages):
                msg.set_metadata("fragment_id", fragment_id)
                msg.set_metadata("fragment_count", len(output_messages))
                msg.set_metadata("fragment_index", idx)

            logger.info(
                "PdfScatterStage.on_data: Fragmented PDF into %d parts with fragment_id=%s",
                len(output_messages),
                fragment_id,
            )
        else:
            logger.debug("PdfScatterStage.on_data: No fragmentation needed, returning original message")

        return output_messages

    def get_fragment_info(self) -> dict:
        """
        Get information about the fragmentation configuration.

        Returns
        -------
        dict
            Configuration details for this stage
        """
        return {
            "stage_type": "PdfScatterStage",
            "pages_per_fragment": self.validated_config.pages_per_fragment,
            "add_overlap": self.validated_config.add_overlap,
            "overlap_pages": self.validated_config.overlap_pages,
            "max_fragments": self.validated_config.max_fragments,
            "metadata": self.validated_config.metadata,
        }
