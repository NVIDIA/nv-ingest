# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
import queue


class PythonStage(ABC):
    """
    Abstract base class for a Python stage in a processing pipeline.

    Simplified version focusing on on_data method with room for future queue.Queue integration.
    """

    def __init__(self, config: BaseModel) -> None:
        self.config: BaseModel = config

        # Reserved for future queue-based processing
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
        """
        pass

    def _processing_loop(self) -> None:
        """
        Stub for future processing loop implementation.
        Reserved for queue-based processing.
        """
        self._logger.debug("Processing loop stub - not yet implemented")
        # Future queue-based implementation would go here

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
