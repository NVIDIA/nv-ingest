# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import ray
import ray.actor
from pydantic import BaseModel
import logging

from ray import get_runtime_context


def setup_stdout_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    return logger


@ray.remote
def external_monitor_actor_shutdown(actor_handle: "RayActorStage", poll_interval: float = 0.1) -> bool:
    """
    Polls the provided actor's `is_shutdown_complete` method until it returns True
    or the actor becomes unreachable.
    """
    logger = setup_stdout_logging("_external_monitor_actor_shutdown")  # Optional: for monitor's own logs

    if actor_handle is None:
        logger.error("Received null actor_handle. Cannot monitor shutdown.")
        return False  # Or raise error

    actor_id_to_monitor = None
    try:
        # Try to get a string representation for logging, might fail if already gone
        actor_id_to_monitor = str(actor_handle)  # Basic representation
    except Exception:
        actor_id_to_monitor = "unknown_actor"

    logger.debug(f"Monitoring shutdown for actor: {actor_id_to_monitor}")

    while True:
        try:
            # Remotely call the actor's method
            if ray.get(actor_handle.is_shutdown_complete.remote()):
                logger.debug(f"Actor {actor_id_to_monitor} reported shutdown complete.")
                actor_handle.request_actor_exit.remote()

                return True
        except ray.exceptions.RayActorError:
            # Actor has died or is otherwise unreachable.
            # Consider this as shutdown complete for the purpose of the future.
            logger.warning(f"Actor {actor_id_to_monitor} became unreachable (RayActorError). Assuming shutdown.")
            return True
        except Exception as e:
            # Catch other potential errors during the remote call
            logger.error(f"Unexpected error while polling shutdown status for {actor_id_to_monitor}: {e}")
            # Depending on policy, either continue polling or assume failure
            return True  # Or True if any exit is "shutdown"

        time.sleep(poll_interval)


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
    _input_queue : Optional[Any]
        Handle to the Ray queue from which input items are read.
        Expected to be set via `set_input_queue`.
    _output_queue : Optional[Any]
        Handle to the Ray queue where processed items are placed.
        Expected to be set via `set_output_queue`.
    _running : bool
        Flag indicating if the processing loop should be actively running.
        Set to True by `start()` and False by `stop()`. Controls the main loop.
    _active_processing : bool
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

    def __init__(self, config: BaseModel, log_to_stdout=False) -> None:
        """
        Initialize the RayActorStage.

        Parameters
        ----------
        config : BaseModel
            Configuration object specific to the stage's behavior. Passed by
            the orchestrator during actor creation.
        """
        self.config: BaseModel = config
        self._input_queue: Optional[Any] = None  # Ray Queue handle expected
        self._output_queue: Optional[Any] = None  # Ray Queue handle expected
        self._running: bool = False
        self._active_processing: bool = False

        # --- Core statistics ---
        self.stats: Dict[str, int] = {
            "active_processing": False,
            "delta_processed": 0,
            "elapsed": 0.0,
            "errors": 0,
            "failed": 0,
            "processed": 0,
            "processing_rate_cps": 0.0,
            "successful_queue_reads": 0,
            "successful_queue_writes": 0,
            "queue_full": 0,
        }
        self.start_time: Optional[float] = None

        # --- State for processing rate calculation ---
        self._last_processed_count: int = 0
        self._last_stats_time: Optional[float] = None

        # --- Threading and shutdown management ---
        self._processing_thread: Optional[threading.Thread] = None
        self._shutting_down: bool = False

        # Lock specifically for coordinating the final shutdown sequence (_request_actor_exit)
        self._lock = threading.Lock()
        self._shutdown_signal_complete = False  # Initialize flag
        self._shutdown_future: Optional[ray.ObjectRef] = None

        # --- Logging ---
        # Ray won't propagate logging to the root logger by default, so we set up a custom logger for debugging
        self._logger = setup_stdout_logging(self.__class__.__name__) if log_to_stdout else logging.getLogger(__name__)

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

    def _read_input(self) -> Optional[Any]:
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
        if not self._running:
            return None

        # Ensure the input queue has been configured before attempting to read
        if self._input_queue is None:
            # This check should ideally not fail if start() is called after setup
            if self._running:
                self._logger.error(f"{self._get_actor_id_str()}: Input queue not set while running")
                # Indicate a programming error - queue should be set before starting
                raise ValueError("Input queue not set while running")
            return None  # Should not happen if self._running is False, but defensive check

        try:
            # Perform a non-blocking or short-blocking read from the queue
            # The timeout allows the loop to check self._running periodically
            return self._input_queue.get(timeout=1.0)
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
        """Core processing routine executed in a dedicated background thread.

        This loop performs the primary work of the actor:
        1.  Continuously attempts to retrieve a `control_message` from the
            `_input_queue`.
        2.  If a message is obtained, it's processed by the `on_data` method.
        3.  If `on_data` yields a result (`updated_cm`), this result is
            indefinitely retried to be `put` onto the `_output_queue`.
        4.  The loop continues as long as the `self._running` flag is `True`.
            This flag is typically controlled by external calls to `start()`
            and `stop()` methods of the actor.
        5.  Upon exiting the main `while` loop (i.e., when `self._running`
            becomes `False`), this method sets `self._shutdown_signal_complete`
            to `True`, indicating to external monitors that the actor's
            processing work is finished.

        Error Handling
        --------------
        - Exceptions raised during `on_data` or the `_output_queue.put`
          sequence are caught, logged, and relevant error statistics are
          incremented. The loop then continues to the next iteration if
          `self._running` is still `True`.
        - If `on_data` returns `None`, it's treated as a recoverable incident;
          a warning is logged, stats are updated, and the loop continues.
          No output is produced for that specific input message.
        - A critical failure in the `_output_queue.put` (e.g., `RayActorError`
          if the queue actor is dead) will currently lead to indefinite retries.

        Statistics
        ----------
        This method updates various keys in `self.stats`, including:
        - `successful_queue_reads`: Incremented when an item is successfully
          read from the input queue.
        - `errors`: Incremented if `on_data` returns `None` or if an
          exception occurs during `on_data` or output queuing.
        - `processed`: Incremented after processing a control message
        - `successful_queue_writes`: Incremented when an item is successfully
          put onto the output queue.
        - `queue_full`: Incremented when an attempt to put to the output
          queue fails (e.g., due to being full or other transient errors),
          triggering a retry.

        Notes
        -----
        - The `self._active_processing` flag is managed to reflect whether
          the `on_data` method is currently (or about to be) active.
        - This method is intended to be the target of a `threading.Thread`.
        - Thread safety for `self.stats` relies on the GIL for simple
          increment operations
        """
        actor_id_str = self._get_actor_id_str()
        self._logger.debug(f"{actor_id_str}: Processing loop thread starting.")

        try:
            while self._running:
                control_message: Optional[Any] = None
                try:
                    # Step 1: Attempt to get work from the input queue.
                    # _read_input() is expected to handle its own timeouts and
                    # return None if no message is available or if self._running became False.
                    control_message = self._read_input()

                    if control_message is None:
                        # No message from input queue (e.g., timeout or shutting down)
                        # Loop back to check self._running again.
                        continue
                    # else: # Implicitly, control_message is not None here
                    self.stats["successful_queue_reads"] += 1

                    # Mark as busy only when a message is retrieved and about to be processed.
                    self._active_processing = True

                    # Step 2: Process the retrieved message using subclass-specific logic.
                    updated_cm: Optional[Any] = self.on_data(control_message)

                    # If there's a valid result and an output queue is configured, attempt to put.
                    if self._output_queue is not None:
                        # This loop will retry indefinitely until the item is put successfully
                        # or an unrecoverable error occurs (which is not explicitly handled to break here).
                        # TODO(Devin) -- This can be improved, should probably fail at some point?
                        #                Consider max retries or specific error handling for RayActorError
                        #                to prevent indefinite blocking if the queue actor is permanently dead.
                        is_put_successful = False
                        while not is_put_successful:  # Renamed loop variable for clarity
                            try:
                                self._output_queue.put(updated_cm)
                                self.stats["successful_queue_writes"] += 1
                                is_put_successful = True  # Exit retry loop on success
                            except Exception as e_put:  # Broad exception catch for put failures
                                self._logger.warning(
                                    f"[{actor_id_str}] Output queue put failed (e.g., full, "
                                    f"timeout, or actor error), retrying. Error: {e_put}"
                                )
                                self.stats["queue_full"] += 1  # Consider renaming if it catches more than "full"
                                time.sleep(0.1)  # Brief pause before retrying

                    # Step 3: Increment "processed" count after successful processing and output (if any).
                    # This is the primary path for "successful processing".
                    self.stats["processed"] += 1

                except Exception as e_item_processing:
                    # Catch exceptions from on_data() or unexpected issues in the item handling block.
                    cm_info_str = f" (message type: {type(control_message).__name__})" if control_message else ""
                    self._logger.exception(
                        f"[{actor_id_str}] Error during processing of item{cm_info_str}: {e_item_processing}"
                    )
                    self.stats["errors"] += 1

                    # If still running, pause briefly to prevent rapid spinning on persistent errors.
                    if self._running:
                        time.sleep(0.1)
                finally:
                    # Ensure _active_processing is reset after each item attempt (success, failure, or no item).
                    self._active_processing = False

            # --- Loop Exit Condition Met ---
            # This point is reached when self._running becomes False.
            self._logger.debug(f"[{actor_id_str}] Graceful exit: self._running is False. Processing loop terminating.")

        except Exception as e_outer_loop:
            # Catches very unexpected errors in the structure of the while loop itself.
            self._logger.exception(
                f"[{actor_id_str}] Unexpected critical error caused processing loop termination: {e_outer_loop}"
            )
        finally:
            # This block executes when the processing thread is about to exit,
            # either due to self._running becoming False or an unhandled critical exception.
            self._logger.debug(f"[{actor_id_str}] Processing loop thread finished.")
            # Signal that this actor's processing duties are complete.
            # External monitors (e.g., via a future from stop()) can use this signal.
            self._shutdown_signal_complete = True

    @staticmethod
    @ray.remote
    def _immediate_true() -> bool:
        """
        A tiny remote method that immediately returns True.
        Used to create a resolved ObjectRef when shutdown is already complete.
        """
        return True

    @ray.method(num_returns=1)
    def _finalize_shutdown(self) -> None:
        """
        Internal Ray method called remotely by the processing thread to safely exit the actor.

        This method runs in the main Ray actor thread context. It acquires a lock
        to prevent multiple exit attempts and then calls `ray.actor.exit_actor()`
        to terminate the actor process gracefully.

        Note: Only necessary if running in a detached actor context.
        """

        actor_id_str = self._get_actor_id_str()
        with self._lock:
            if self._shutting_down:
                return

            self._shutting_down = True

        self._logger.info(f"{actor_id_str}: Executing actor exit process.")

        get_runtime_context().current_actor.request_actor_exit.remote()

    @ray.method(num_returns=1)
    def request_actor_exit(self) -> None:
        """
        Request the actor to exit gracefully.

        This method is called from the main Ray actor thread to ensure a clean
        shutdown of the actor. It should be called when the processing loop
        has completed its work and is ready to exit.
        """

        if self._processing_thread:
            self._processing_thread.join()

        self._shutdown_signal_complete = True

        self._logger.debug(f"{self._get_actor_id_str()}: Requesting actor exit.")
        ray.actor.exit_actor()

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
        if self._running:
            self._logger.warning(f"{actor_id_str}: Start called but actor is already running.")
            return False

        self._logger.info(f"{actor_id_str}: Starting actor...")
        # --- Initialize Actor State ---
        self._running = True
        self._shutting_down = False  # Reset shutdown flag on start
        self._shutdown_signal_complete = False
        self.start_time = time.time()

        # --- Reset Statistics ---
        self._last_stats_time = self.start_time
        self._last_processed_count = 0

        # --- Start Background Processing Thread ---
        self._logger.debug(f"{actor_id_str}: Creating and starting processing thread.")
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=False,
        )
        self._processing_thread.start()

        self._logger.info(f"{actor_id_str}: Actor started successfully.")

        return True

    @ray.method(num_returns=1)
    def stop(self) -> ray.ObjectRef:
        actor_id_str = self._get_actor_id_str()
        self._logger.info(f"{actor_id_str}: Received external stop request.")

        if self._shutdown_future is not None:
            self._logger.debug(f"{actor_id_str}: Stop called again, returning existing shutdown future.")
            return self._shutdown_future

        if not self._running and self._shutdown_signal_complete:  # Check if already fully shutdown
            self._logger.info(f"{actor_id_str}: Stop called, but actor was already shutdown and signal complete.")
            if self._shutdown_future:  # Should have been set by the previous shutdown sequence
                return self._shutdown_future
            else:  # Should not happen if shutdown_signal_complete is true, but as a fallback
                self._shutdown_future = self._immediate_true.remote()
                return self._shutdown_future
        elif not self._running:  # Was stopped but maybe not fully signaled (e.g. mid-shutdown)
            self._logger.warning(
                f"{actor_id_str}: Stop called but actor was not running (or already stopping). "
                "Will create/return monitor future."
            )
            # If _shutdown_future is None here, it means stop wasn't called before OR a previous
            # monitor didn't get stored. Proceed to create a new monitor.
            # If it *was* already stopping and _shutdown_future exists, the first `if` catches it.

        # --- Initiate Shutdown signal to internal loop (if still running) ---
        if self._running:  # Only set self._running = False if it was actually running
            self._running = False
            self._logger.info(f"{actor_id_str}: Stop signal sent to processing loop. Shutdown initiated.")
        else:
            self._logger.info(
                f"{actor_id_str}: Actor processing loop was already stopped. Monitoring for final shutdown signal."
            )

        # --- Spawn shutdown watcher task ---
        # Get a handle to the current actor instance to pass to the monitor.
        # This is crucial: the monitor needs to call methods on *this specific actor*.
        try:
            self_handle = get_runtime_context().current_actor
        except Exception as e:
            self._logger.error(
                f"{actor_id_str}: Failed to get current_actor handle for monitoring: {e}. Returning a failing future."
            )

            # Cannot proceed to monitor, return a future that resolves to False or raises
            @ray.remote
            def failed_future():
                raise RuntimeError("Failed to initiate shutdown monitoring due to missing actor handle.")

            return failed_future.remote()  # Or ray.put(False) directly

        self._shutdown_future = external_monitor_actor_shutdown.remote(self_handle)

        return self._shutdown_future

    @ray.method(num_returns=1)
    def is_shutdown_complete(self) -> bool:
        """
        Checks if the actor's processing loop has finished and signaled completion.
        Raises RayActorError if the actor process has terminated.
        """
        return self._shutdown_signal_complete

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
        current_processed: int = self.stats.get("processed", 0)
        is_active: bool = self._active_processing
        delta_processed = 0

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
            "active_processing": is_active,  # Return the state captured at the beginning
            "delta_processed": delta_processed,
            "elapsed": elapsed,
            "errors": self.stats.get("errors", 0),
            "failed": self.stats.get("failed", 0),
            "processed": current_processed,
            "processing_rate_cps": processing_rate_cps,
            "queue_full": self.stats.get("queue_full", 0),
            "successful_queue_reads": self.stats.get("successful_queue_reads", 0),
            "successful_queue_writes": self.stats.get("successful_queue_writes", 0),
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
        self._logger.debug(f"{self._get_actor_id_str()}: Setting input queue.")
        self._input_queue = queue_handle
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
        self._logger.debug(f"{self._get_actor_id_str()}: Setting output queue.")
        self._output_queue = queue_handle
        return True
