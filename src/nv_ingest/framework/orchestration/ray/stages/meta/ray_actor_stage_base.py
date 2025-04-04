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

from ray import get_runtime_context

logger = logging.getLogger(__name__)


class RayActorStage(ABC):
    """
    Abstract base class for a Ray actor stage in a pipeline.

    Utilizes a background thread for processing and ensures actor exit
    is called safely from the main Ray thread context upon completion,
    using the runtime context to get the actor handle.
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
        """
        # Actor name is no longer needed here
        self.config: BaseModel = config
        self.progress_engine_count: int = progress_engine_count
        self.input_queue: Optional[Any] = None
        self.output_queue: Optional[Any] = None
        self.running: bool = False
        self.active_processing: bool = False
        self.stats: Dict[str, int] = {"processed": 0}
        self.start_time: Optional[float] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._shutting_down: bool = False  # Flag to prevent multiple exit calls
        self._lock = threading.Lock()  # Lock for thread-safe access to _shutting_down
        # Store actor ID upon init for clearer logging, obtained from runtime context
        try:
            # Get context once during init if possible (might not be fully set yet)
            # It's safer to get it dynamically in logs/methods when needed.
            # self._actor_id_str = f"Actor {get_runtime_context().get_actor_id()}"
            pass  # Decide if storing ID is useful or just get dynamically
        except Exception:
            # self._actor_id_str = "Actor (ID unavailable at init)"
            pass

    def _get_actor_id_str(self) -> str:
        """Helper to safely get actor ID string for logging."""
        try:
            return f"Actor {get_runtime_context().get_actor_id()}"
        except Exception:
            return "Actor (ID unavailable)"

    def read_input(self) -> Any:
        """
        Read a control message from the input queue. Non-blocking after timeout.
        """
        if not self.running:
            return None
        if self.input_queue is None:
            if self.running:
                # Log with actor context if possible
                logger.error(f"{self._get_actor_id_str()}: Input queue not set while running")
                raise ValueError("Input queue not set while running")
            return None
        try:
            return self.input_queue.get(timeout=1.0)
        except Exception:
            return None

    @abstractmethod
    def on_data(self, control_message: Any) -> Any:
        """
        Process a control message. Must be implemented by subclasses.
        """
        pass

    def _processing_loop(self) -> None:
        """
        Internal processing loop running in a background thread.

        Upon finishing, it requests the main actor thread to exit using the
        runtime context to get the actor handle.
        """
        actor_id_str = self._get_actor_id_str()  # Get ID for logging within the thread
        logger.debug(f"{actor_id_str}: Processing loop starting.")
        try:
            while self.running:
                control_message: Any = None
                try:
                    control_message = self.read_input()
                    if control_message is None:
                        continue

                    self.active_processing = True
                    updated_cm: Any = self.on_data(control_message)
                    if updated_cm and self.output_queue is not None:
                        if self.running:
                            self.output_queue.put(updated_cm)
                        else:
                            logger.warning(f"{actor_id_str}: Suppressed output write after stop signal.")
                    self.stats["processed"] += 1
                except Exception as e:
                    cm_info = f" (message type: {type(control_message).__name__})" if control_message else ""
                    logger.exception(f"{actor_id_str}: Error in processing loop{cm_info}: {e}")
                    if self.running:
                        time.sleep(0.1)
                finally:
                    self.active_processing = False
            logger.debug(f"{actor_id_str}: Running flag is false, exiting loop.")
        except Exception as e:
            logger.exception(f"{actor_id_str}: Unexpected error in processing loop scope: {e}")
        finally:
            logger.debug(f"{actor_id_str}: Processing loop finished.")
            # --- Trigger exit from the main thread using runtime context ---
            try:
                # Get a handle to this actor using the runtime context
                logger.debug(f"{actor_id_str}: Requesting actor exit via runtime context handle.")
                # This works even when called from the background thread
                self_handle = get_runtime_context().current_actor
                if self_handle:
                    # Schedule the _request_actor_exit method to run on the main thread
                    self_handle._request_actor_exit.remote()
                else:
                    # This case should be rare within an actor context
                    logger.error(f"{actor_id_str}: Could not get current_actor handle from runtime context.")

            except Exception as e:
                # Catch potential errors getting context or scheduling the call
                logger.exception(f"{actor_id_str}: Error requesting actor exit via runtime context: {e}")

    @ray.method(num_returns=1)
    def _request_actor_exit(self) -> None:
        """
        Internal Ray method called to safely exit the actor. Runs in main context.
        """
        actor_id_str = self._get_actor_id_str()
        with self._lock:
            if self._shutting_down:
                # logger.warning(f"{actor_id_str}: Exit already requested, ignoring duplicate call.")
                return  # Reduce noise, just return if already shutting down
            self._shutting_down = True

        logger.info(f"{actor_id_str}: Exit requested by internal call. Shutting down actor.")
        try:
            ray.actor.exit_actor()
        except Exception as e:
            logger.critical(f"{actor_id_str}: CRITICAL - Error during ray.actor.exit_actor(): {e}", exc_info=True)

    @ray.method(num_returns=1)
    def start(self) -> bool:
        """
        Start the processing loop in a separate daemon thread.
        """
        actor_id_str = self._get_actor_id_str()
        if self.running:
            logger.warning(f"{actor_id_str}: Start called but already running.")
            return False
        logger.info(f"{actor_id_str}: Starting...")
        self.running = True
        self._shutting_down = False  # Reset flag
        self.start_time = time.time()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info(f"{actor_id_str}: Started.")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        """
        Signals the processing loop to stop. Non-blocking.
        """
        actor_id_str = self._get_actor_id_str()
        logger.info(f"{actor_id_str}: Stop requested externally.")

        if not self.running:
            logger.warning(f"{actor_id_str}: Stop called but not running.")
            return False

        self.running = False
        logger.info(f"{actor_id_str}: Stop signal sent to processing loop.")
        return True

    # --- Other methods (get_stats, set_input_queue, set_output_queue) ---

    @ray.method(num_returns=1)
    def get_stats(self) -> Dict[str, Any]:
        """Retrieve processing statistics."""
        elapsed: float = time.time() - self.start_time if self.start_time else 0
        return {"processed": self.stats["processed"], "elapsed": elapsed, "active_processing": self.active_processing}

    @ray.method(num_returns=1)
    def set_input_queue(self, queue_handle: Any) -> bool:
        """Set the input queue handle for the stage."""
        self.input_queue = queue_handle
        return True

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: Any) -> bool:
        """Set the output queue handle for the stage."""
        self.output_queue = queue_handle
        return True
