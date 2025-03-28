# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import ray
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class RayActorStage(ABC):
    """
    Abstract base class for a Ray actor stage in a pipeline.

    This class provides the framework for reading input from a queue, processing data,
    and writing output to another queue. Concrete implementations must override the
    on_data() method to provide custom data processing logic.
    """

    def __init__(self, config: BaseModel, progress_engine_count: int = 1) -> None:
        """
        Initialize the RayActorStage.

        Parameters
        ----------
        config : BaseModel
            Configuration for the stage.
        progress_engine_count : int, optional
            Number of progress engine threads to run, by default 1

        Attributes
        ----------
        input_queue : Optional[Any]
            Queue handle for input; initially None.
        output_queue : Optional[Any]
            Queue handle for output; initially None.
        running : bool
            Flag indicating whether the stage is running.
        processing_complete : bool
            Flag indicating whether the processing loop has completed.
        active_processing : bool
            Flag indicating if a job is currently being processed.
        stats : dict
            Dictionary storing processing statistics.
        start_time : Optional[float]
            Timestamp when processing was started.
        """
        self.config: BaseModel = config
        self.progress_engine_count: int = progress_engine_count
        self.input_queue: Optional[Any] = None
        self.output_queue: Optional[Any] = None
        self.running: bool = False
        self.processing_complete: bool = False
        self.active_processing: bool = False
        self.stats: Dict[str, int] = {"processed": 0}
        self.start_time: Optional[float] = None

    def read_input(self) -> Any:
        """
        Read a control message from the input queue.

        Returns
        -------
        Any
            The control message from the input queue, or None if not running or on timeout.

        Raises
        ------
        ValueError
            If the input queue is not set.
        """
        if not self.running:
            return None
        if self.input_queue is None:
            raise ValueError("Input queue not set")
        try:
            # Directly get from the queue with a timeout.
            return self.input_queue.get(timeout=10.0)
        except Exception:
            # logger.exception("Timeout or error reading from input queue: %s", e)
            return None

    @abstractmethod
    def on_data(self, control_message: Any) -> Any:
        """
        Process a control message.

        Parameters
        ----------
        control_message : Any
            The control message to process.

        Returns
        -------
        Any
            The processed control message.
        """
        pass

    def _processing_loop(self) -> None:
        """
        Internal processing loop that continuously reads from the input queue,
        processes the data, and writes the result to the output queue.

        The loop sets the active_processing flag while processing each message.
        """
        try:
            while self.running:
                try:
                    control_message: Any = self.read_input()
                    if control_message is None:
                        continue
                    # Mark that we are actively processing this job.
                    self.active_processing = True
                    updated_cm: Any = self.on_data(control_message)
                    if updated_cm and self.output_queue is not None:
                        # Put the updated control message into the output queue directly.
                        self.output_queue.put(updated_cm)
                    self.stats["processed"] += 1
                except Exception as e:
                    logger.exception("Error in processing loop: %s", e)
                    time.sleep(1.0)
                finally:
                    # Reset active_processing after each job.
                    self.active_processing = False
        finally:
            if not self.running:
                # logger.debug("Processing loop detected self.running is False; exiting loop.")
                pass
            self.processing_complete = True
            # logger.debug("Processing loop has set processing_complete to True.")
            ray.actor.exit_actor()

    @ray.method(num_returns=1)
    def start(self) -> bool:
        """
        Start the processing loop in a separate daemon thread.

        Returns
        -------
        bool
            True if the stage started successfully, False if it was already running.
        """
        if self.running:
            return False
        self.running = True
        self.processing_complete = False
        self.start_time = time.time()
        threading.Thread(target=self._processing_loop, daemon=True).start()
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        """
        Stop the processing loop.

        This method sets the running flag to False and waits until the processing loop
        has completed.

        Returns
        -------
        bool
            True when the stage has stopped.
        """
        self.running = False
        while not self.processing_complete:
            # logger.debug("Waiting for processing loop to complete...")
            time.sleep(2.0)
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve processing statistics.

        Returns
        -------
        dict
            A dictionary containing the number of processed messages, elapsed time,
            and active processing state.
        """
        elapsed: float = time.time() - self.start_time if self.start_time else 0
        return {"processed": self.stats["processed"], "elapsed": elapsed, "active_processing": self.active_processing}

    @ray.method(num_returns=1)
    def set_input_queue(self, queue_handle: Any) -> bool:
        """
        Set the input queue handle for the stage.

        Parameters
        ----------
        queue_handle : Any
            The queue to be used for input.

        Returns
        -------
        bool
            True after the input queue has been set.
        """
        self.input_queue = queue_handle
        return True

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: Any) -> bool:
        """
        Set the output queue handle for the stage.

        Parameters
        ----------
        queue_handle : Any
            The queue to be used for output.

        Returns
        -------
        bool
            True after the output queue has been set.
        """
        self.output_queue = queue_handle
        return True
