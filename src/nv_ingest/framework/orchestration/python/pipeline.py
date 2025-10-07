# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import List, Callable, Any, Optional
from datetime import datetime

from nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
)
from nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
)

logger = logging.getLogger(__name__)


class PythonPipeline:
    """
    Simple Python-based pipeline orchestrator.

    This orchestrator runs a chain of Python functions between a source and sink,
    without Ray dependencies. It's designed for simple, local processing.
    """

    def __init__(
        self,
        source: PythonMessageBrokerTaskSource,
        sink: PythonMessageBrokerTaskSink,
        processing_functions: Optional[List[Callable[[Any], Any]]] = None,
    ):
        """
        Initialize the pipeline.

        Parameters
        ----------
        source : PythonMessageBrokerTaskSource
            The source to read messages from.
        sink : PythonMessageBrokerTaskSink
            The sink to write processed messages to.
        processing_functions : List[Callable[[Any], Any]], optional
            List of functions to apply to messages in sequence.
            If None, messages pass through unchanged (no-op pipeline).
        """
        self.source = source
        self.sink = sink
        self.processing_functions = processing_functions or []
        self._running = False
        self._processed_count = 0
        self._error_count = 0
        self.start_time = None

        logger.info(f"PythonPipeline initialized with {len(self.processing_functions)} processing functions")

    def _process_message(self, message: Any) -> Any:
        """
        Apply all processing functions to a message in sequence.

        Parameters
        ----------
        message : Any
            The input message to process.

        Returns
        -------
        Any
            The processed message after applying all functions.
        """
        processed_message = message

        for i, func in enumerate(self.processing_functions):
            try:
                logger.debug(f"Applying processing function {i+1}/{len(self.processing_functions)}")
                processed_message = func(processed_message)
            except Exception as e:
                logger.error(f"Processing function {i+1} failed: {e}")
                raise

        return processed_message

    def run_single_iteration(self) -> bool:
        """
        Run a single iteration of the pipeline.

        Returns
        -------
        bool
            True if a message was processed, False if no message was available.
        """
        try:
            # Get message from source
            message = self.source.get_message()

            if message is None:
                return False

            logger.debug(f"Processing message: {message.message_id}")

            # Process the message through the function chain
            processed_message = self._process_message(message)

            # Send to sink
            success = self.sink.process_message(processed_message)

            if success:
                self._processed_count += 1
                logger.debug(f"Successfully processed message: {message.message_id}")
            else:
                self._error_count += 1
                logger.error(f"Failed to process message: {message.message_id}")

            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Pipeline iteration failed: {e}")
            return False

    def run(self, max_iterations: Optional[int] = None, poll_interval: float = 0.1) -> None:
        """
        Run the pipeline continuously.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations to run. If None, runs indefinitely.
        poll_interval : float
            Time to sleep between iterations when no messages are available.
        """
        self.start()

        iteration_count = 0

        try:
            while self._running:
                if max_iterations is not None and iteration_count >= max_iterations:
                    logger.info(f"Reached maximum iterations: {max_iterations}")
                    break

                # Run single iteration
                message_processed = self.run_single_iteration()

                if not message_processed:
                    # No message available, sleep briefly
                    time.sleep(poll_interval)

                iteration_count += 1

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.stop()

    def start(self) -> None:
        """Start the pipeline and its components."""
        self._running = True
        self.start_time = datetime.now()

        # Start source and sink
        self.source.start()
        self.sink.start()

        logger.info("PythonPipeline started")

    def stop(self) -> None:
        """Stop the pipeline and its components."""
        self._running = False

        # Stop source and sink
        self.source.stop()
        self.sink.stop()

        logger.info("PythonPipeline stopped")

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "running": self._running,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "source_stats": self.source.get_stats(),
            "sink_stats": self.sink.get_stats(),
        }
