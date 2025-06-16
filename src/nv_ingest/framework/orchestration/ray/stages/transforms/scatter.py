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
        self._logger.info(f"LambdaScatterStage.on_data: Received message: {message}")
        if hasattr(message, "get_metadata") and callable(message.get_metadata):
            self._logger.info(
                f"LambdaScatterStage.on_data: Incoming message metadata: {message.get_metadata(key=None)}"
            )

        try:
            self._logger.info(f"LambdaScatterStage.on_data: Applying transform_callable: {self._transform_callable}")
            output_messages = self._transform_callable(message)
            self._logger.info(
                f"LambdaScatterStage.on_data: Transform callable produced {len(output_messages)} output_messages."
            )

            if not isinstance(output_messages, list):
                self._logger.error(
                    f"LambdaScatterStage.on_data: Transform callable did not return a list. "
                    f"Got: {type(output_messages)}"
                )
                raise ValueError(f"Transform callable must return a list, got {type(output_messages)}")

            if len(output_messages) == 0:
                self._logger.error("LambdaScatterStage.on_data: Transform callable returned an empty list.")
                raise ValueError("Transform callable must return at least 1 message")

            for idx, msg in enumerate(output_messages):
                if not isinstance(msg, IngestControlMessage):
                    self._logger.error(
                        f"LambdaScatterStage.on_data: Output message at index {idx} is not an"
                        f" IngestControlMessage. Got: {type(msg)}"
                    )
                    raise ValueError(f"Output message at index {idx} is not an IngestControlMessage")

            if len(output_messages) > 1:
                fragment_id = str(uuid.uuid4())
                self._logger.info(
                    f"LambdaScatterStage.on_data: Scattering detected. "
                    f"generated fragment_id: {fragment_id} for {len(output_messages)} fragments."
                )
                for idx, msg in enumerate(output_messages):
                    msg.set_metadata("fragment_id", fragment_id)
                    msg.set_metadata("fragment_count", len(output_messages))
                    msg.set_metadata("fragment_index", idx)
                    self._logger.info(
                        f"LambdaScatterStage.on_data: Annotated fragment {idx+1}/{len(output_messages)} "
                        f"with fragment metadata."
                    )
            else:
                self._logger.info(
                    "LambdaScatterStage.on_data: No scattering (single output message), skipping fragment annotation."
                )

            self._logger.info(
                f"LambdaScatterStage.on_data: Successfully processed message. "
                f"Returning {len(output_messages)} messages."
            )

        except Exception as e:
            self._logger.error(f"LambdaScatterStage.on_data: Transform callable failed: {str(e)}", exc_info=True)
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
        super().__init__(config, log_to_stdout=True)
        self.validated_config: PDFScatterSchema = config

        # Create the PDF fragmenter with the configured parameters and pass the actor's logger
        self._pdf_fragmenter = create_pdf_fragmenter(
            pages_per_fragment=config.pages_per_fragment, actor_logger=self._logger
        )

        logger.info(
            "PdfScatterStage initialized with pages_per_fragment=%d, overlap=%s, overlap_pages=%d",
            config.pages_per_fragment,
        )

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
        self._logger.info(f"PDFScatterStage.on_data: Received message: {message}")
        try:
            if hasattr(message, "get_metadata") and callable(message.get_metadata):
                self._logger.info(
                    f"PDFScatterStage.on_data: Incoming message metadata: {message.get_metadata(key=None)}"
                )
                if hasattr(message, "payload") and callable(message.payload):
                    payload_df = message.payload()
                    if not payload_df.empty and "document_type" in payload_df.columns:
                        self._logger.info(
                            f"PDFScatterStage.on_data: Incoming message document_type: "
                            f"{payload_df.iloc[0].get('document_type')}"
                        )

            # Apply PDF fragmentation
            self._logger.info("PDFScatterStage.on_data: Applying _pdf_fragmenter")
            output_messages = self._pdf_fragmenter(message)
            self._logger.info(
                f"PDFScatterStage.on_data: _pdf_fragmenter produced {len(output_messages)} output_messages."
            )

            # Validate we don't exceed max fragments
            if len(output_messages) > self.validated_config.max_fragments:
                self._logger.warning(
                    f"PDFScatterStage.on_data: PDF fragmentation produced "
                    f"{len(output_messages)} fragments, exceeding max_fragments={self.validated_config.max_fragments}. "
                    f"Truncating."
                )
                output_messages = output_messages[: self.validated_config.max_fragments]
                self._logger.info(f"PDFScatterStage.on_data: Truncated output_messages to {len(output_messages)}.")

            # Annotate fragments if we have more than one
            if len(output_messages) > 1:
                fragment_id = str(uuid.uuid4())
                self._logger.info(
                    f"PDFScatterStage.on_data: PDF scattering detected. "
                    f"Generated fragment_id: {fragment_id} for {len(output_messages)} fragments."
                )
                for idx, msg in enumerate(output_messages):
                    self._logger.info(
                        f"PDFScatterStage.on_data: LOOP START for fragment {idx+1}/{len(output_messages)}. "
                        f"Msg ID (if available): {getattr(msg, 'id', 'N/A')}, Msg type: {type(msg)}"
                    )
                    if hasattr(msg, "set_metadata") and callable(msg.set_metadata):
                        self._logger.info(f"  Attempting to set fragment_id for {idx+1}")
                        msg.set_metadata("fragment_id", fragment_id)
                        self._logger.info(f"  Successfully set fragment_id for {idx+1}")

                        self._logger.info(f"  Attempting to set fragment_count for {idx+1}")
                        msg.set_metadata("fragment_count", len(output_messages))
                        self._logger.info(f"  Successfully set fragment_count for {idx+1}")

                        self._logger.info(f"  Attempting to set fragment_index for {idx+1}")
                        msg.set_metadata("fragment_index", idx)
                        self._logger.info(f"  Successfully set fragment_index for {idx+1}")

                        self._logger.info(
                            f"PDFScatterStage.on_data: Annotated fragment {idx+1}/{len(output_messages)} "
                            f"with fragment metadata."
                        )
                    else:
                        self._logger.error(
                            f"PDFScatterStage.on_data: Cannot annotate fragment {idx+1}/{len(output_messages)}. "
                            f"Item is not a valid message object (type: {type(msg)}). "
                            f"Skipping annotation for this item."
                        )

                self._logger.info(
                    f"PDFScatterStage.on_data: Finished annotating {len(output_messages)} "
                    f"fragments with fragment_id={fragment_id}"
                )
            else:
                self._logger.info(
                    "PDFScatterStage.on_data: No PDF fragmentation needed or resulted in single output, "
                    "skipping fragment group annotation."
                )

            self._logger.info(
                f"PDFScatterStage.on_data: Successfully processed message. Returning {len(output_messages)} messages."
            )
            return output_messages
        except Exception as e:
            self._logger.error(f"PDFScatterStage.on_data: An unexpected error occurred: {str(e)}", exc_info=True)
            # Re-raise the exception so that the @nv_ingest_node_failure_try_except decorator can handle it
            # or it propagates if not caught by that decorator.
            raise

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
