# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import threading
from typing import Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

from .stages.meta.python_stage_base import PythonStage

logger = logging.getLogger(__name__)


class PythonPipeline:
    """
    Simple Python-based pipeline orchestrator.

    This orchestrator runs a linear chain of stages between a source and sink,
    without Ray dependencies. It's designed for simple, local processing with
    background execution capability.

    Simplified design constraints:
    - Only one source allowed
    - Only one sink allowed
    - Multiple stages allowed in linear order
    - No queues (direct function chaining)
    - Background execution when started
    """

    def __init__(self):
        """Initialize the Python pipeline."""
        self._source: Optional[Any] = None
        self._sink: Optional[Any] = None
        self._stages: List[PythonStage] = []

        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._processed_count = 0
        self._error_count = 0
        self.start_time: Optional[datetime] = None

        logger.info("PythonPipeline initialized")

    def add_source(self, *, name: str, source_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a source stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the source stage.
        source_actor : Any
            The source instance (must have get_message method).
        config : BaseModel
            The configuration for the source stage.

        Returns
        -------
        PythonPipeline
            The pipeline instance for method chaining.

        Raises
        ------
        ValueError
            If a source is already added.
        """
        if self._source is not None:
            raise ValueError("Only one source is allowed in PythonPipeline")

        self._source = source_actor
        logger.info(f"Added source stage: {name}")
        return self

    def add_sink(self, *, name: str, sink_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a sink stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the sink stage.
        sink_actor : Any
            The sink instance (must have process_message method).
        config : BaseModel
            The configuration for the sink stage.

        Returns
        -------
        PythonPipeline
            The pipeline instance for method chaining.

        Raises
        ------
        ValueError
            If a sink is already added.
        """
        if self._sink is not None:
            raise ValueError("Only one sink is allowed in PythonPipeline")

        self._sink = sink_actor
        logger.info(f"Added sink stage: {name}")
        return self

    def add_stage(self, *, name: str, stage_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a processing stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the stage.
        stage_actor : Any
            The stage instance (must inherit from PythonStage with on_data method).
        config : BaseModel
            The configuration for the stage.

        Returns
        -------
        PythonPipeline
            The pipeline instance for method chaining.
        """
        if not isinstance(stage_actor, PythonStage):
            raise ValueError(f"Stage {name} must inherit from PythonStage")

        self._stages.append(stage_actor)
        logger.info(f"Added processing stage: {name} (total stages: {len(self._stages)})")
        return self

    def build(self) -> None:
        """
        Build the pipeline and validate configuration.

        Raises
        ------
        ValueError
            If source or sink is missing.
        """
        if self._source is None:
            raise ValueError("Pipeline must have a source")
        if self._sink is None:
            raise ValueError("Pipeline must have a sink")

        # Start all stages
        if hasattr(self._source, "start"):
            self._source.start()
        if hasattr(self._sink, "start"):
            self._sink.start()
        for stage in self._stages:
            stage.start()

        logger.info(f"Pipeline built with 1 source, {len(self._stages)} stages, 1 sink")

    def _process_single_message(self) -> bool:
        """
        Process a single message through the pipeline.

        Returns
        -------
        bool
            True if a message was processed, False if no message available.
        """
        try:
            # Get message from source
            logger.debug("Pipeline attempting to get message from source")
            message = self._source.get_message()

            if message is None:
                logger.debug("Pipeline: No message received from source")
                return False

            logger.info(f"Pipeline received message: {getattr(message, 'message_id', 'unknown')}")

            # Process through all stages in linear order
            current_message = message
            for i, stage in enumerate(self._stages):
                try:
                    current_message = stage.on_data(current_message)
                    if current_message is None:
                        logger.warning(f"Stage {i} returned None, stopping processing")
                        self._error_count += 1
                        return True

                    # Update stage statistics
                    stage.stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Error in stage {i}: {e}")
                    stage.stats["errors"] += 1
                    self._error_count += 1
                    return True

            # Send to sink
            try:
                result = self._sink.on_data(current_message)
                success = result is not None
            except Exception as e:
                logger.error(f"Error in sink: {e}")
                success = False

            if success:
                self._processed_count += 1
                logger.info("Pipeline successfully processed message")
            else:
                self._error_count += 1
                logger.error("Failed to process message in sink")

            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Pipeline processing failed: {e}")
            return False

    def _processing_loop(self) -> None:
        """
        Main processing loop that runs in a background thread.

        Continuously processes messages until stopped.
        """
        logger.info("Pipeline processing loop started")

        while self._running:
            try:
                # Process a single message
                logger.debug("Pipeline processing loop iteration")
                message_processed = self._process_single_message()

                if not message_processed:
                    # No message available, sleep briefly to avoid busy waiting
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)  # Sleep longer on error

        logger.info("Pipeline processing loop stopped")

    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Start the pipeline in a background thread.

        Parameters
        ----------
        monitor_poll_interval : float
            Unused, kept for interface compatibility.
        scaling_poll_interval : float
            Unused, kept for interface compatibility.
        """
        if self._running:
            logger.warning("Pipeline is already running")
            return

        # Build pipeline if not already built
        if not hasattr(self._source, "start") or not self._source._running:
            self.build()

        self._running = True
        self.start_time = datetime.now()

        # Reset statistics
        self._processed_count = 0
        self._error_count = 0

        # Start the processing loop in a background thread
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

        logger.info("PythonPipeline started in background thread")

    def stop(self) -> None:
        """
        Stop the pipeline and perform cleanup.
        """
        if not self._running:
            logger.warning("Pipeline is not running")
            return

        self._running = False

        # Wait for processing thread to finish
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)

        # Stop all components
        if hasattr(self._source, "stop"):
            self._source.stop()
        if hasattr(self._sink, "stop"):
            self._sink.stop()
        for stage in self._stages:
            stage.stop()

        logger.info("PythonPipeline stopped")

    def get_stats(self) -> dict:
        """
        Get pipeline statistics.

        Returns
        -------
        dict
            Dictionary containing pipeline statistics.
        """
        elapsed = 0.0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        processing_rate = 0.0
        if elapsed > 0:
            processing_rate = self._processed_count / elapsed

        stats = {
            "running": self._running,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "elapsed_seconds": elapsed,
            "processing_rate_cps": processing_rate,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "source_stats": getattr(self._source, "get_stats", lambda: {})(),
            "sink_stats": getattr(self._sink, "get_stats", lambda: {})(),
            "stage_stats": [stage.get_stats() for stage in self._stages],
        }

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.stop()
