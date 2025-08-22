# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import threading
import multiprocessing
from typing import Any, Optional, List, Dict, Tuple
from datetime import datetime
from pydantic import BaseModel

from .stages.meta.python_stage_base import PythonStage

logger = logging.getLogger(__name__)


class PipelineEdge:
    """Represents a directed edge between two stages in the pipeline."""

    def __init__(self, from_stage: str, to_stage: str, queue_size: int = 1000):
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.queue = multiprocessing.Queue(maxsize=queue_size)
        self.queue_size = queue_size


class PythonPipeline:
    """
    Python-based pipeline orchestrator with streaming asynchronous execution.

    This orchestrator supports both linear (legacy) and graph-based pipeline execution
    using multiprocessing for true parallelism. Each stage runs in its own process
    and communicates via multiprocessing queues.

    Features:
    - Linear pipeline execution (backward compatible)
    - Graph-based pipeline with add_edge method
    - Multiprocessing-based stage execution
    - Queue-based message passing between stages
    - Graceful shutdown and lifecycle management
    """

    def __init__(self, enable_streaming: bool = True):
        """
        Initialize the Python pipeline.

        Args:
            enable_streaming: If True, enables streaming asynchronous execution with multiprocessing.
                             If False, uses legacy linear synchronous execution.
        """
        # Legacy linear pipeline components
        self._source: Optional[Any] = None
        self._sink: Optional[Any] = None
        self._stages: List[PythonStage] = []

        # Streaming pipeline components
        self.enable_streaming = enable_streaming
        self._stage_registry: Dict[str, Tuple[Any, BaseModel]] = {}  # name -> (stage_class, config)
        self._edges: List[PipelineEdge] = []
        self._processes: Dict[str, multiprocessing.Process] = {}
        self._stage_queues: Dict[str, Dict[str, multiprocessing.Queue]] = {}  # stage_name -> {input/output: queue}

        # Pipeline state
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._processed_count = 0
        self._error_count = 0
        self.start_time: Optional[datetime] = None

        # Broker management for proper cleanup
        self._broker_instance: Optional[Any] = None
        self._broker_thread: Optional[threading.Thread] = None

        logger.info(f"PythonPipeline initialized (streaming={'enabled' if enable_streaming else 'disabled'})")

    def set_broker_instance(self, broker_instance: Any, broker_thread: Optional[threading.Thread] = None) -> None:
        """
        Set the broker instance for proper cleanup during pipeline shutdown.

        Args:
            broker_instance: The SimpleBroker instance
            broker_thread: Optional broker thread for cleanup
        """
        self._broker_instance = broker_instance
        self._broker_thread = broker_thread
        logger.info("Broker instance registered with pipeline for cleanup")

    def add_source(self, *, name: str, source_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a source stage to the pipeline.

        Args:
            name: Name of the source stage
            source_actor: Source stage class or instance
            config: Configuration for the source stage

        Returns:
            Self for method chaining
        """
        if self.enable_streaming:
            # For streaming mode, we need the class, not the instance
            if hasattr(source_actor, "__class__") and not isinstance(source_actor, type):
                # It's an instance, store the class and config
                self._stage_registry[name] = (source_actor.__class__, config)
            else:
                # It's a class, store as-is
                self._stage_registry[name] = (source_actor, config)
            logger.info(f"Added streaming source stage: {name}")
        else:
            # Legacy behavior - handle both classes and instances
            if isinstance(source_actor, type):
                # It's a class, instantiate it
                self._source = source_actor(config=config, stage_name=name)
            else:
                # It's already an instance, use as-is
                self._source = source_actor

            if self._source is not None and hasattr(self._source, "_source"):
                # Check if we already have a source
                logger.warning("Multiple sources detected in linear pipeline mode")

            logger.info(f"Added linear source stage: {name}")

        return self

    def add_sink(self, *, name: str, sink_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a sink stage to the pipeline.

        Args:
            name: Name of the sink stage
            sink_actor: Sink stage class or instance
            config: Configuration for the sink stage

        Returns:
            Self for method chaining
        """
        if self.enable_streaming:
            # For streaming mode, we need the class, not the instance
            if hasattr(sink_actor, "__class__") and not isinstance(sink_actor, type):
                # It's an instance, store the class and config
                self._stage_registry[name] = (sink_actor.__class__, config)
            else:
                # It's a class, store as-is
                self._stage_registry[name] = (sink_actor, config)
            logger.info(f"Added streaming sink stage: {name}")
        else:
            # Legacy behavior - handle both classes and instances
            if isinstance(sink_actor, type):
                # It's a class, instantiate it
                self._sink = sink_actor(config=config, stage_name=name)
            else:
                # It's already an instance, use as-is
                self._sink = sink_actor

            if self._sink is not None and hasattr(self._sink, "_sink"):
                # Check if we already have a sink
                logger.warning("Multiple sinks detected in linear pipeline mode")

            logger.info(f"Added linear sink stage: {name}")

        return self

    def add_stage(self, *, name: str, stage_actor: Any, config: BaseModel) -> "PythonPipeline":
        """
        Adds a processing stage to the pipeline.

        Args:
            name: Name of the stage
            stage_actor: Stage class or instance
            config: Configuration for the stage

        Returns:
            Self for method chaining
        """
        if self.enable_streaming:
            # For streaming mode, we need the class, not the instance
            if hasattr(stage_actor, "__class__") and not isinstance(stage_actor, type):
                # It's an instance, store the class and config
                self._stage_registry[name] = (stage_actor.__class__, config)
            else:
                # It's a class, store as-is
                self._stage_registry[name] = (stage_actor, config)
            logger.info(f"Added streaming stage: {name}")
        else:
            # Legacy behavior - handle both classes and instances
            if isinstance(stage_actor, type):
                # It's a class, instantiate it
                stage_instance = stage_actor(config=config, stage_name=name)
            else:
                # It's already an instance, use as-is
                stage_instance = stage_actor

            self._stages.append(stage_instance)
            logger.info(f"Added linear stage: {name}")

        return self

    def add_edge(self, from_stage: str, to_stage: str, queue_size: int = 1000) -> "PythonPipeline":
        """
        Adds a directed edge between two stages in the streaming pipeline.

        Args:
            from_stage: Name of the source stage
            to_stage: Name of the destination stage
            queue_size: Maximum size of the queue between stages

        Returns:
            Self for method chaining

        Raises:
            ValueError: If streaming is not enabled or stages don't exist
        """
        if not self.enable_streaming:
            raise ValueError("add_edge is only available when streaming is enabled")

        if from_stage not in self._stage_registry:
            raise ValueError(f"Source stage '{from_stage}' not found in pipeline")

        if to_stage not in self._stage_registry:
            raise ValueError(f"Destination stage '{to_stage}' not found in pipeline")

        edge = PipelineEdge(from_stage, to_stage, queue_size)
        self._edges.append(edge)

        logger.info(f"Added edge: {from_stage} -> {to_stage} (queue_size={queue_size})")
        return self

    def _setup_stage_queues(self) -> None:
        """Set up input and output queues for each stage based on edges."""
        # Initialize queue dictionaries for each stage
        for stage_name in self._stage_registry:
            self._stage_queues[stage_name] = {"inputs": [], "outputs": []}

        # Assign queues based on edges
        for edge in self._edges:
            # Add output queue to source stage
            self._stage_queues[edge.from_stage]["outputs"].append(edge.queue)
            # Add input queue to destination stage
            self._stage_queues[edge.to_stage]["inputs"].append(edge.queue)

    def _run_stage_process(self, stage_name: str, stage_class: Any, config: BaseModel) -> None:
        """
        Run a single stage in its own process with queue-based message passing.

        Args:
            stage_name: Name of the stage
            stage_class: Stage class to instantiate
            config: Configuration for the stage
        """
        try:
            # Create stage instance
            stage_instance = stage_class(config=config, stage_name=stage_name)

            # Get input and output queues for this stage
            input_queues = self._stage_queues[stage_name]["inputs"]
            output_queues = self._stage_queues[stage_name]["outputs"]

            logger.info(f"Stage {stage_name} started with {len(input_queues)} inputs, {len(output_queues)} outputs")

            # Set up stage queues
            stage_instance._input_queues = input_queues
            stage_instance._output_queues = output_queues

            # Start the stage's processing loop
            stage_instance._processing_loop()

        except Exception as e:
            logger.error(f"Error in stage {stage_name}: {e}")
            raise

    def start(self) -> None:
        """Start the pipeline execution."""
        if self._running:
            logger.warning("Pipeline is already running")
            return

        self._running = True
        self.start_time = datetime.now()

        if self.enable_streaming:
            self._start_streaming_pipeline()
        else:
            self._start_linear_pipeline()

    def _start_streaming_pipeline(self) -> None:
        """Start the streaming pipeline with multiprocessing."""
        logger.info("Starting streaming pipeline with multiprocessing")

        # Validate pipeline configuration
        if not self._stage_registry:
            raise ValueError("No stages configured for streaming pipeline")

        if not self._edges:
            logger.warning("No edges configured - stages will run independently")

        # Set up queues between stages
        self._setup_stage_queues()

        # Validate that all stages have proper queue connections
        for stage_name in self._stage_registry:
            input_count = len(self._stage_queues[stage_name]["inputs"])
            output_count = len(self._stage_queues[stage_name]["outputs"])
            logger.info(f"Stage '{stage_name}': {input_count} inputs, {output_count} outputs")

        # Start each stage in its own process
        for stage_name, (stage_class, config) in self._stage_registry.items():
            try:
                process = multiprocessing.Process(
                    target=self._run_stage_process, args=(stage_name, stage_class, config), name=f"Stage-{stage_name}"
                )
                process.start()
                self._processes[stage_name] = process
                logger.info(f"Started process for stage: {stage_name} (PID: {process.pid})")
            except Exception as e:
                logger.error(f"Failed to start stage {stage_name}: {e}")
                # Clean up any already started processes
                self._stop_streaming_pipeline()
                raise

    def _start_linear_pipeline(self) -> None:
        """Start the legacy linear pipeline with threading."""
        logger.info("Starting linear pipeline with threading")

        if self._source is None:
            raise ValueError("No source stage configured")
        if self._sink is None:
            raise ValueError("No sink stage configured")

        self._processing_thread = threading.Thread(target=self._linear_processing_loop, daemon=True)
        self._processing_thread.start()

    def _linear_processing_loop(self) -> None:
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

    def stop(self) -> None:
        """Stop the pipeline execution gracefully."""
        if not self._running:
            logger.warning("Pipeline is not running")
            return

        logger.info("Stopping pipeline...")
        self._running = False

        if self.enable_streaming:
            self._stop_streaming_pipeline()
        else:
            self._stop_linear_pipeline()

        # Calculate final statistics
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info(
                f"Pipeline stopped. Processed: {self._processed_count}, "
                f"Errors: {self._error_count}, Elapsed: {elapsed:.2f}s"
            )

    def _stop_streaming_pipeline(self) -> None:
        """Stop the streaming pipeline and terminate all processes."""
        logger.info("Stopping streaming pipeline processes...")

        # Send termination signals to all queues
        for edge in self._edges:
            try:
                edge.queue.put(None, timeout=1.0)  # Sentinel value to stop processing
            except:  # noqa
                pass  # Queue might be full or closed

        # Wait for processes to terminate gracefully
        for stage_name, process in self._processes.items():
            try:
                process.join(timeout=5.0)
                if process.is_alive():
                    logger.warning(f"Force terminating stage: {stage_name}")
                    process.terminate()
                    process.join(timeout=2.0)
                    if process.is_alive():
                        logger.error(f"Failed to terminate stage: {stage_name}")
                else:
                    logger.info(f"Stage {stage_name} terminated gracefully")
            except Exception as e:
                logger.error(f"Error stopping stage {stage_name}: {e}")

        self._processes.clear()

        # Stop broker instance if registered
        if self._broker_instance:
            try:
                logger.info("Shutting down broker instance...")
                self._broker_instance.shutdown()
                logger.info("Broker instance shutdown complete")
            except Exception as e:
                logger.error(f"Failed to shutdown broker instance: {e}")

        # Stop broker thread if registered
        if self._broker_thread and self._broker_thread.is_alive():
            try:
                logger.info("Waiting for broker thread to stop...")
                self._broker_thread.join(timeout=5.0)
                if self._broker_thread.is_alive():
                    logger.warning("Broker thread did not stop gracefully within timeout")
                else:
                    logger.info("Broker thread stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping broker thread: {e}")

    def _stop_linear_pipeline(self) -> None:
        """Stop the legacy linear pipeline."""
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            if self._processing_thread.is_alive():
                logger.warning("Processing thread did not stop gracefully")

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
            "streaming_enabled": self.enable_streaming,
        }

        if self.enable_streaming:
            # Add streaming-specific statistics
            stats.update(
                {
                    "active_processes": len([p for p in self._processes.values() if p.is_alive()]),
                    "total_processes": len(self._processes),
                    "edges_count": len(self._edges),
                    "stages_count": len(self._stage_registry),
                    "process_stats": {
                        name: {
                            "pid": process.pid if process.is_alive() else None,
                            "is_alive": process.is_alive(),
                            "exitcode": process.exitcode,
                        }
                        for name, process in self._processes.items()
                    },
                }
            )
        else:
            # Add linear pipeline statistics
            stats.update(
                {
                    "source_stats": getattr(self._source, "get_stats", lambda: {})(),
                    "sink_stats": getattr(self._sink, "get_stats", lambda: {})(),
                    "stage_stats": [stage.get_stats() for stage in self._stages],
                }
            )

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.stop()
