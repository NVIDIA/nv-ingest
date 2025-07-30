# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
import queue
import multiprocessing


class PythonStage(ABC):
    """
    Abstract base class for a Python stage in a processing pipeline.

    Supports both synchronous processing (on_data method) and asynchronous
    multiprocessing-based streaming (processing loop with queues).
    """

    def __init__(self, config: BaseModel) -> None:
        self.config: BaseModel = config

        # Queue-based processing for streaming pipelines
        self._input_queues: Optional[List[multiprocessing.Queue]] = None
        self._output_queues: Optional[List[multiprocessing.Queue]] = None

        # Legacy single queue support (backward compatibility)
        self._input_queue: Optional[queue.Queue] = None
        self._output_queue: Optional[queue.Queue] = None

        self._running: bool = False
        self.stats: Dict[str, Any] = {
            "processed": 0,
            "errors": 0,
            "elapsed": 0.0,
        }
        self.start_time: Optional[float] = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def on_data(self, control_message: Any) -> Optional[Any]:
        """
        Process a single data item (control message).
        Must be implemented by subclasses.

        Args:
            control_message: The message to process

        Returns:
            Processed message or None if no output
        """
        pass

    def _processing_loop(self) -> None:
        """
        Main processing loop for multiprocessing-based streaming execution.

        Continuously reads messages from input queues, processes them via on_data,
        and sends results to output queues. Supports multiple input and output queues.
        """
        self._logger.info(f"Starting processing loop for {self.__class__.__name__}")
        self._running = True
        self.start_time = time.time()

        try:
            while self._running:
                message_processed = False

                # Check all input queues for messages
                if self._input_queues:
                    for input_queue in self._input_queues:
                        try:
                            # Non-blocking get with short timeout
                            message = input_queue.get(timeout=0.1)

                            # Check for termination sentinel
                            if message is None:
                                self._logger.info("Received termination signal")
                                self._running = False
                                break

                            # Process the message
                            result = self._process_message_with_stats(message)

                            # Send result to all output queues
                            if result is not None and self._output_queues:
                                for output_queue in self._output_queues:
                                    try:
                                        output_queue.put(result, timeout=1.0)
                                    except Exception as e:
                                        # Handle both queue.Full and multiprocessing queue exceptions
                                        self._logger.warning(f"Output queue full or error, dropping message: {e}")
                                        self.stats["errors"] += 1

                            message_processed = True

                        except Exception as e:
                            # Handle both queue.Empty and multiprocessing queue exceptions
                            if "Empty" in str(type(e).__name__):
                                continue  # Try next queue
                            else:
                                self._logger.error(f"Error processing message: {e}")
                                self.stats["errors"] += 1

                # If no input queues configured, this stage should wait or be a true source
                else:
                    # Only call source generation if this stage explicitly supports it
                    # Most stages should just wait for input
                    if hasattr(self, "_is_source_stage") and self._is_source_stage:
                        try:
                            result = self._generate_source_message()
                            if result is not None and self._output_queues:
                                for output_queue in self._output_queues:
                                    try:
                                        output_queue.put(result, timeout=1.0)
                                        message_processed = True
                                    except Exception as e:
                                        self._logger.warning(f"Output queue full or error, dropping message: {e}")
                                        self.stats["errors"] += 1
                            elif result is None:
                                # Source has no more messages, sleep to avoid busy waiting
                                time.sleep(0.1)
                        except Exception as e:
                            self._logger.error(f"Error generating source message: {e}")
                            self.stats["errors"] += 1
                    else:
                        # Regular processing stage with no input queues - just wait
                        time.sleep(0.1)

                # If no message was processed, sleep briefly to avoid busy waiting
                if not message_processed:
                    time.sleep(0.01)

        except KeyboardInterrupt:
            self._logger.info("Processing loop interrupted")
        except Exception as e:
            self._logger.error(f"Fatal error in processing loop: {e}")
        finally:
            self._running = False
            self._logger.info(f"Processing loop stopped. Stats: {self.stats}")

    def _process_message_with_stats(self, message: Any) -> Optional[Any]:
        """
        Process a message and update statistics.

        Args:
            message: The message to process

        Returns:
            Processed message or None
        """
        try:
            start_time = time.time()
            result = self.on_data(message)

            # Update statistics
            self.stats["processed"] += 1
            self.stats["elapsed"] += time.time() - start_time

            return result

        except Exception as e:
            self._logger.error(f"Error in on_data: {e}")
            self.stats["errors"] += 1
            return None

    def _generate_source_message(self) -> Optional[Any]:
        """
        Generate a message for source stages.

        This method should only be called by stages that explicitly mark themselves as source stages.
        Source stages MUST override this method to provide their own message generation logic.

        Returns:
            Generated message or None

        Raises:
            NotImplementedError: If source stage doesn't implement this method
        """
        raise NotImplementedError(
            f"Source stage {self.__class__.__name__} must implement _generate_source_message() method. "
            f"This method should return a message to send downstream, or None when no more messages are available."
        )

    def mark_as_source_stage(self) -> None:
        """
        Mark this stage as a source stage that generates messages.

        Source stages should call this method in their __init__ to indicate
        they generate messages rather than just process input.
        """
        self._is_source_stage = True

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.start_time is not None:
            self.stats["elapsed"] = time.time() - self.start_time
        return self.stats.copy()

    def start(self) -> bool:
        """Start the stage."""
        if self._running:
            return False

        self._running = True
        self.start_time = time.time()
        self.stats["processed"] = 0
        self.stats["errors"] = 0

        self._logger.info(f"Stage {self.__class__.__name__} started")
        return True

    def stop(self) -> None:
        """Stop the stage."""
        self._running = False
        self._logger.info(f"Stage {self.__class__.__name__} stopped")

    def is_running(self) -> bool:
        """Check if stage is running."""
        return self._running

    # Future queue methods (reserved for later)
    def set_input_queue(self, queue_handle: queue.Queue) -> bool:
        """Set input queue for future queue-based processing."""
        self._input_queue = queue_handle
        return True

    def set_output_queue(self, queue_handle: queue.Queue) -> bool:
        """Set output queue for future queue-based processing."""
        self._output_queue = queue_handle
        return True
