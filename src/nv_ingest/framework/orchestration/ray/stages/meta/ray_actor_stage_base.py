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
    Abstract base class for a stateful Ray actor stage in a processing pipeline.

    This class provides a common structure for actors that consume items from
    an input queue, process them, and potentially place results onto an output
    queue. It utilizes a background thread for the main processing loop to
    avoid blocking the main Ray actor thread. It includes basic statistics
    tracking (processed count, elapsed time, processing rate) and mechanisms
    for graceful shutdown.

    Subclasses must implement the `on_data` method to define the specific
    processing logic for each item.

    Attributes
    ----------
    config : BaseModel
        Configuration object for the stage.
    input_queue : Optional[Any]
        Handle to the Ray queue from which input items are read.
        Expected to be set via `set_input_queue`.
    output_queue : Optional[Any]
        Handle to the Ray queue where processed items are placed.
        Expected to be set via `set_output_queue`.
    running : bool
        Flag indicating if the processing loop should be actively running.
        Set to True by `start()` and False by `stop()`. Controls the main loop.
    active_processing : bool
        Flag indicating if the `on_data` method is currently executing.
        Useful for understanding if the actor is busy at a given moment.
    stats : Dict[str, int]
        Dictionary to store basic operational statistics. Currently tracks 'processed'.
    start_time : Optional[float]
        Timestamp (from time.time()) when the `start()` method was called.
        Used for calculating total elapsed time.
    _last_processed_count : int
        Internal state variable storing the processed count at the last `get_stats` call.
        Used for calculating interval processing rate.
    _last_stats_time : Optional[float]
        Internal state variable storing the timestamp of the last `get_stats` call.
        Used for calculating interval processing rate.
    _processing_thread : Optional[threading.Thread]
        Handle to the background thread running the `_processing_loop`.
    _shutting_down : bool
        Internal flag to prevent redundant shutdown actions, protected by _lock.
    _lock : threading.Lock
        Lock to protect access to shutdown-related state (`_shutting_down`).
    """

    def __init__(self, config: BaseModel) -> None:
        """
        Initialize the RayActorStage.

        Parameters
        ----------
        config : BaseModel
            Configuration object specific to the stage's behavior. Passed by
            the orchestrator during actor creation.
        """
        self.config: BaseModel = config
        self.input_queue: Optional[Any] = None  # Ray Queue handle expected
        self.output_queue: Optional[Any] = None  # Ray Queue handle expected
        self.running: bool = False
        self.active_processing: bool = False

        # --- Core statistics ---
        self.stats: Dict[str, int] = {"processed": 0}
        self.start_time: Optional[float] = None

        # --- State for processing rate calculation ---
        self._last_processed_count: int = 0
        self._last_stats_time: Optional[float] = None

        # --- Threading and shutdown management ---
        self._processing_thread: Optional[threading.Thread] = None
        self._shutting_down: bool = False
        # Lock specifically for coordinating the final shutdown sequence (_request_actor_exit)
        self._lock = threading.Lock()

    @staticmethod
    def _get_actor_id_str() -> str:
        """
        Helper method to safely get the current Ray actor ID string for logging.

        Handles cases where the runtime context or actor ID might not be available.

        Returns
        -------
        str
            A formatted string representing the actor ID or a fallback message.
        """
        try:
            # Attempt to get the full actor ID from Ray's runtime context
            return f"Actor {get_runtime_context().get_actor_id()}"
        except Exception:
            # Fallback if running outside a Ray actor context or if context fails
            return "Actor (ID unavailable)"

    def read_input(self) -> Optional[Any]:
        """
        Reads an item from the input queue with a timeout.

        This method attempts to get an item from the configured `input_queue`.
        It uses a timeout to prevent indefinite blocking, allowing the
        processing loop to remain responsive to the `running` flag.

        Returns
        -------
        Optional[Any]
            The item read from the queue, or None if the queue is empty after
            the timeout, the queue is not set, or the actor is not running.

        Raises
        ------
        ValueError
            If `input_queue` is None while the actor's `running` flag is True.
            This indicates a configuration error.
        """
        # Do not attempt to read if the actor has been signaled to stop
        if not self.running:
            return None

        # Ensure the input queue has been configured before attempting to read
        if self.input_queue is None:
            # This check should ideally not fail if start() is called after setup
            if self.running:
                logger.error(f"{self._get_actor_id_str()}: Input queue not set while running")
                # Indicate a programming error - queue should be set before starting
                raise ValueError("Input queue not set while running")
            return None  # Should not happen if self.running is False, but defensive check

        try:
            # Perform a non-blocking or short-blocking read from the queue
            # The timeout allows the loop to check self.running periodically
            return self.input_queue.get(timeout=1.0)
        except Exception:
            # Common exceptions include queue.Empty in older Ray versions or
            # custom queue implementations raising timeout errors.
            # Return None to signify no item was retrieved this cycle.
            return None

    @abstractmethod
    def on_data(self, control_message: Any) -> Optional[Any]:
        """
        Process a single data item (control message).

        This is the core logic method that must be implemented by subclasses.
        It receives an item dequeued by `read_input` and performs the
        stage-specific processing.

        Parameters
        ----------
        control_message : Any
            The data item retrieved from the input queue.

        Returns
        -------
        Optional[Any]
            The result of the processing. If a result is returned (not None),
            it will be placed onto the `output_queue`. Return None if this
            stage does not produce output or if this specific message yields
            no result.
        """
        pass  # Must be implemented by concrete subclasses

    def _processing_loop(self) -> None:
        """
        The main processing loop executed in a background thread.

        Continuously reads from the input queue, processes items using `on_data`,
        and puts results onto the output queue. Exits when `self.running` becomes
        False. Upon loop termination, it schedules `_request_actor_exit` to run
        on the main Ray actor thread to ensure a clean shutdown via `ray.actor.exit_actor()`.
        """
        actor_id_str = self._get_actor_id_str()
        logger.debug(f"{actor_id_str}: Processing loop thread starting.")

        try:
            # Loop continues as long as the actor is marked as running
            while self.running:
                control_message: Optional[Any] = None
                try:
                    # Step 1: Attempt to get work from the input queue
                    control_message = self.read_input()

                    # If no message, loop back and check self.running again
                    if control_message is None:
                        continue  # Go to the next iteration of the while loop

                    # Step 2: Process the retrieved message
                    self.active_processing = True  # Mark as busy
                    updated_cm: Optional[Any] = self.on_data(control_message)
                    # Note: self.active_processing is set to False in the finally block

                    # Step 3: Handle the output
                    if updated_cm is not None and self.output_queue is not None:
                        try:
                            self.output_queue.put(updated_cm)
                        except Exception as put_err:
                            # Log errors during put, especially relevant if the queue
                            # or downstream actors are also shutting down.
                            logger.error(
                                f"{actor_id_str}: Error putting result to output queue: {put_err}", exc_info=True
                            )

                    # Step 4: Increment processed count (thread safety note)
                    self.stats["processed"] += 1

                except Exception as e:
                    # Log exceptions during item processing but continue the loop
                    cm_info = f" (message type: {type(control_message).__name__})" if control_message else ""
                    logger.exception(f"{actor_id_str}: Error processing item{cm_info}: {e}")
                    # Avoid busy-spinning in case of persistent errors reading or processing
                    if self.running:
                        time.sleep(0.1)
                finally:
                    # Ensure active_processing is reset regardless of success/failure/output
                    self.active_processing = False

            # --- Loop Exit ---
            logger.debug(
                f"{actor_id_str}: Graceful exit condition met (self.running is False). Processing loop terminating."
            )

        except Exception as e:
            # Catch unexpected errors in the loop structure itself
            logger.exception(f"{actor_id_str}: Unexpected error caused processing loop termination: {e}")
        finally:
            logger.debug(f"{actor_id_str}: Processing loop thread finished.")
            # --- Trigger Actor Exit from Main Thread ---
            # It's crucial to call ray.actor.exit_actor() from the main actor
            # thread, not the background thread. We use the current_actor handle
            # obtained via the runtime context to schedule the exit call remotely
            # (but targeting the same actor).
            try:
                logger.debug(f"{actor_id_str}: Scheduling final actor exit via _request_actor_exit.")
                # Get a handle to the current actor instance
                self_handle = get_runtime_context().current_actor
                if self_handle:
                    # Asynchronously call the _request_actor_exit method on this actor.
                    # Ray ensures this method runs on the main actor thread.
                    self_handle._request_actor_exit.remote()
                else:
                    # This should generally not happen if called from within an actor method/thread.
                    logger.error(
                        f"{actor_id_str}: Could not obtain current_actor handle. Actor might not exit cleanly."
                    )
            except Exception as e:
                # Log errors during the scheduling of the exit call
                logger.exception(f"{actor_id_str}: Failed to schedule _request_actor_exit: {e}")

    @ray.method(num_returns=1)
    def _request_actor_exit(self) -> None:
        """
        Internal Ray method called remotely by the processing thread to safely exit the actor.

        This method runs in the main Ray actor thread context. It acquires a lock
        to prevent multiple exit attempts and then calls `ray.actor.exit_actor()`
        to terminate the actor process gracefully.
        """
        actor_id_str = self._get_actor_id_str()
        # Use a lock to ensure exit logic runs only once, even if triggered multiple times
        with self._lock:
            if self._shutting_down:
                logger.warning(f"{actor_id_str}: Exit already in progress, ignoring redundant request.")
                return
            # Mark that shutdown has been initiated
            self._shutting_down = True

        logger.info(f"{actor_id_str}: Executing actor exit process.")
        try:
            # The official way to stop the actor process from within
            ray.actor.exit_actor()
            # This call does not return; the actor process terminates.
        except Exception as e:
            logger.critical(
                f"{actor_id_str}: " f"CRITICAL - Failed to execute ray.actor.exit_actor(): {e}", exc_info=True
            )

    @ray.method(num_returns=1)
    def start(self) -> bool:
        """
        Starts the actor's processing loop in a background thread.

        Initializes state, resets statistics, and launches the `_processing_loop`
        thread. Idempotent: if called while already running, it logs a warning
        and returns False.

        Returns
        -------
        bool
            True if the actor was successfully started, False if it was already running.
        """
        actor_id_str = self._get_actor_id_str()
        # Prevent starting if already running
        if self.running:
            logger.warning(f"{actor_id_str}: Start called but actor is already running.")
            return False

        logger.info(f"{actor_id_str}: Starting actor...")
        # --- Initialize Actor State ---
        self.running = True
        self._shutting_down = False  # Reset shutdown flag on start
        self.start_time = time.time()

        # --- Reset Statistics ---
        self.stats["processed"] = 0
        # Initialize rate calculation timers and counts
        self._last_stats_time = self.start_time
        self._last_processed_count = 0

        # --- Start Background Processing Thread ---
        logger.debug(f"{actor_id_str}: Creating and starting processing thread.")
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,  # Set as daemon so it doesn't block Python exit if main thread dies unexpectedly
        )
        self._processing_thread.start()

        logger.info(f"{actor_id_str}: Actor started successfully.")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        """
        Signals the actor's processing loop to stop gracefully.

        Sets the `running` flag to False. The background processing thread will
        detect this flag, finish its current task (if any), and then initiate
        the actor exit sequence via `_request_actor_exit`. This method returns
        immediately and does not wait for the actor to fully shut down.

        Returns
        -------
        bool
            True if the stop signal was sent (i.e., the actor was running),
            False if the actor was already stopped.
        """
        actor_id_str = self._get_actor_id_str()
        logger.info(f"{actor_id_str}: Received external stop request.")

        # Check if the actor is actually running
        if not self.running:
            logger.warning(f"{actor_id_str}: Stop called but actor was not running.")
            return False

        # Signal the processing loop to stop by setting the flag
        self.running = False
        logger.info(f"{actor_id_str}: Stop signal sent to processing loop. Shutdown initiated.")

        # Note: The actual termination happens asynchronously when the loop finishes.
        return True

    # --- get_stats ---

    @ray.method(num_returns=1)
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves performance statistics for the actor.

        Calculates the approximate processing rate since the last call to
        `get_stats` or since `start()`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing statistics:
              - 'processed' (int): Total items processed since the actor started.
              - 'elapsed' (float): Total time in seconds since the actor started.
              - 'active_processing' (bool): Whether the actor was actively
                                            processing an item in `on_data`
                                            at the moment this method was called.
              - 'processing_rate_cps' (float): Calculated items processed per
                                               second during the last interval.
                                               Can be zero if no items were
                                               processed or the interval was too short.
        """
        current_time: float = time.time()
        current_processed: int = self.stats["processed"]
        is_active: bool = self.active_processing

        processing_rate_cps: float = 0.0  # Default rate

        # Calculate rate only if actor has started and stats have been initialized
        if self._last_stats_time is not None and self.start_time is not None:
            delta_time: float = current_time - self._last_stats_time
            # Use the processed count captured at the start of this method call
            delta_processed: int = current_processed - self._last_processed_count

            # Calculate rate if time has advanced and items were processed
            # Use a small epsilon for delta_time to avoid division by zero
            if delta_time > 0.001 and delta_processed >= 0:
                processing_rate_cps = delta_processed / delta_time
            # If delta_processed is negative (e.g., due to counter reset or race), report 0 rate.

        # Update state for the *next* interval calculation AFTER computing the current rate
        self._last_stats_time = current_time
        self._last_processed_count = current_processed  # Store the count used in *this* interval calculation

        # Calculate total elapsed time
        elapsed: float = (current_time - self.start_time) if self.start_time else 0.0

        # Compile and return the statistics dictionary
        return {
            "processed": current_processed,
            "elapsed": elapsed,
            "active_processing": is_active,  # Return the state captured at the beginning
            "processing_rate_cps": processing_rate_cps,
        }

    @ray.method(num_returns=1)
    def set_input_queue(self, queue_handle: Any) -> bool:
        """
        Sets the input queue handle for this actor stage.

        Should be called before `start()`.

        Parameters
        ----------
        queue_handle : Any
            The Ray queue handle (e.g., `ray.util.queue.Queue`) from which
            this actor should read input items.

        Returns
        -------
        bool
            True indicating the queue was set.
        """
        logger.debug(f"{self._get_actor_id_str()}: Setting input queue.")
        self.input_queue = queue_handle
        return True

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: Any) -> bool:
        """
        Sets the output queue handle for this actor stage.

        Should be called before `start()`.

        Parameters
        ----------
        queue_handle : Any
            The Ray queue handle (e.g., `ray.util.queue.Queue`) to which
            this actor should write output items.

        Returns
        -------
        bool
            True indicating the queue was set.
        """
        logger.debug(f"{self._get_actor_id_str()}: Setting output queue.")
        self.output_queue = queue_handle
        return True
