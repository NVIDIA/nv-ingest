# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os
import psutil

import ray
import ray.actor
from pydantic import BaseModel
import logging
import pyarrow as pa

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
    stage_name : Optional[str]
        Name of the stage from YAML pipeline configuration. Used by
        stage-aware decorators for consistent naming.
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

    def __init__(self, config: BaseModel, stage_name: Optional[str] = None, log_to_stdout=False) -> None:
        """
        Initialize the RayActorStage.

        Parameters
        ----------
        config : BaseModel
            Configuration object specific to the stage's behavior. Passed by
            the orchestrator during actor creation.
        stage_name : Optional[str]
            Name of the stage from YAML pipeline configuration. Used by
            stage-aware decorators for consistent naming.
        log_to_stdout : bool
            Whether to enable stdout logging.
        """
        self.config: BaseModel = config
        self.stage_name: Optional[str] = stage_name
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

        # --- Logging ---
        # Ray won't propagate logging to the root logger by default, so we set up a custom logger for debugging
        self._logger = setup_stdout_logging(self.__class__.__name__) if log_to_stdout else logging.getLogger(__name__)

        self._actor_id_str = self._get_actor_id_str()

        # --- PyArrow Memory Management ---
        # Time-based periodic cleanup to prevent long-term memory accumulation
        self._memory_cleanup_interval_seconds = getattr(
            config, "memory_cleanup_interval_seconds", 300
        )  # 5 minutes default
        self._last_memory_cleanup_time = time.time()
        self._memory_cleanups_performed = 0

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
                self._logger.error(f"{self._actor_id_str}: Input queue not set while running")
                # Indicate a programming error - queue should be set before starting
                raise ValueError("Input queue not set while running")
            return None  # Should not happen if self._running is False, but defensive check

        item: Optional[Any] = None
        try:
            item = self._input_queue.get(timeout=1.0)

            if item is None:
                return None

            if isinstance(item, ray.ObjectRef):
                try:
                    deserialized_object = ray.get(item)
                except ray.exceptions.ObjectLostError:
                    self._logger.error(
                        f"[{self._actor_id_str}] Failed to retrieve object from Ray object store. "
                        f"It has been lost and cannot be recovered."
                    )
                    raise  # Re-raise the exception to be handled by the processing loop

                del item
                return deserialized_object

            return item

        except Exception:
            if item is not None and isinstance(item, ray.ObjectRef):
                del item
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
        - `processed`: Incremented after successful processing and output (if any).
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
        self._logger.debug(f"{self._actor_id_str}: Processing loop thread starting.")

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
                    updated_cm = self.on_data(control_message)

                    # If there's a valid result and an output queue is configured, attempt to put.
                    if self._output_queue is not None and updated_cm is not None:
                        object_ref_to_put = None  # Ensure var exists for the finally block
                        try:
                            # Get the handle of the queue actor to set it as the owner.
                            # This decouples the object's lifetime from this actor.
                            owner_actor = self._output_queue.actor

                            # Put the object into Plasma, transferring ownership.
                            object_ref_to_put = ray.put(updated_cm, _owner=owner_actor)

                            # Now that the object is safely in Plasma, we can delete the large local copy.
                            del updated_cm

                            # This loop will retry until the ObjectRef is put successfully or shutdown is initiated.
                            is_put_successful = False
                            while not is_put_successful:
                                try:
                                    self._output_queue.put(object_ref_to_put)
                                    self.stats["successful_queue_writes"] += 1
                                    is_put_successful = True  # Exit retry loop on success
                                except Exception as e_put:
                                    self._logger.warning(
                                        f"[{self._actor_id_str}] Output queue put failed (e.g., full, "
                                        f"timeout, or actor error), retrying. Error: {e_put}"
                                    )
                                    self.stats["queue_full"] += 1
                                    time.sleep(0.1)  # Brief pause before retrying
                        finally:
                            # After the operation, delete the local ObjectRef.
                            # The primary reference is now held by the queue actor.
                            if object_ref_to_put is not None:
                                del object_ref_to_put

                    # Step 3: Increment "processed" count after successful processing and output (if any).
                    # This is the primary path for "successful processing".
                    self.stats["processed"] += 1

                    # Time-based PyArrow memory cleanup check (best-effort, low overhead)
                    try:
                        current_time = time.time()
                        if (current_time - self._last_memory_cleanup_time) >= self._memory_cleanup_interval_seconds:
                            self._force_arrow_memory_cleanup()
                            self._last_memory_cleanup_time = current_time
                    except Exception:
                        # Never allow cleanup issues to interfere with processing
                        pass

                except ray.exceptions.ObjectLostError:
                    # This error is handled inside the loop to prevent the actor from crashing.
                    # We log it and continue to the next message.
                    self._logger.error(f"[{self._actor_id_str}] CRITICAL: An object was lost in transit. Skipping.")
                    # In a real-world scenario, you might want to increment a metric for monitoring.
                    continue

                except Exception as e_item_processing:
                    # Catch exceptions from on_data() or unexpected issues in the item handling block.
                    cm_info_str = f" (message type: {type(control_message).__name__})" if control_message else ""
                    self._logger.exception(
                        f"[{self._actor_id_str}] Error during processing of item{cm_info_str}: {e_item_processing}"
                    )
                    self.stats["errors"] += 1

                    # If still running, pause briefly to prevent rapid spinning on persistent errors.
                    if self._running:
                        time.sleep(0.1)
                finally:
                    # Ensure _active_processing is reset after each item attempt (success, failure, or no item).
                    self._active_processing = False

                    # Explicitly delete the reference to the control message to aid garbage collection.
                    # This is important for large messages, as it helps release memory and ObjectRefs sooner.
                    if control_message is not None:
                        del control_message

            # --- Loop Exit Condition Met ---
            # This point is reached when self._running becomes False.
            self._logger.debug(
                f"[{self._actor_id_str}] Graceful exit: self._running is False. Processing loop terminating."
            )

        except Exception as e_outer_loop:
            # Catches very unexpected errors in the structure of the while loop itself.
            self._logger.exception(
                f"[{self._actor_id_str}] Unexpected critical error caused processing loop termination: {e_outer_loop}"
            )
        finally:
            # This block executes when the processing thread is about to exit,
            # either due to self._running becoming False or an unhandled critical exception.
            self._logger.debug(f"[{self._actor_id_str}] Processing loop thread finished.")
            # Perform a best-effort final memory cleanup on exit
            try:
                self._force_arrow_memory_cleanup()
            except Exception:
                pass
            # Signal that this actor's processing duties are complete.
            # External monitors (e.g., via a future from stop()) can use this signal.
            self._shutdown_signal_complete = True

    def _force_arrow_memory_cleanup(self) -> None:
        """
        Best-effort memory cleanup for PyArrow allocations.

        - Runs Python garbage collection to drop unreachable references.
        - If PyArrow is available and its default memory pool supports
          release_unused(), request it to return free pages to the OS.

        Designed to be safe to call periodically; any failures are logged at
        debug/warning levels and are non-fatal.
        """
        try:
            # First, trigger Python GC to maximize reclaimable memory
            gc.collect()

            try:
                pool = pa.default_memory_pool()
                try:
                    before_bytes = getattr(pool, "bytes_allocated", lambda: 0)()
                except Exception:
                    before_bytes = 0

                released = False
                if hasattr(pool, "release_unused"):
                    try:
                        pool.release_unused()
                        released = True
                    except Exception as e_release:
                        self._logger.debug(f"[{self._actor_id_str}] Arrow pool release_unused() failed: {e_release}")

                try:
                    after_bytes = getattr(pool, "bytes_allocated", lambda: before_bytes)()
                except Exception:
                    after_bytes = before_bytes

                if released:
                    delta_mb = max(0, (before_bytes - after_bytes) / (1024 * 1024))
                    if delta_mb > 0:
                        self._logger.debug(
                            f"[{self._actor_id_str}] Arrow cleanup released ~{delta_mb:.2f}"
                            f" MB (pool now {after_bytes/(1024*1024):.2f} MB)."
                        )
                self._memory_cleanups_performed += 1
            except ModuleNotFoundError:
                # PyArrow not present; nothing to do beyond GC.
                self._memory_cleanups_performed += 1
            except Exception as e_pa:
                # Any other PyArrow-related issues are non-fatal.
                self._logger.debug(f"[{self._actor_id_str}] Arrow cleanup skipped due to error: {e_pa}")
                self._memory_cleanups_performed += 1
        except Exception as e:
            # As a last resort, swallow any errors to avoid interfering with the actor loop.
            self._logger.debug(f"[{self._actor_id_str}] Memory cleanup encountered an error: {e}")

    def _get_memory_usage_mb(self) -> float:
        """
        Gets the total memory usage of the current actor process (RSS).

        Returns
        -------
        float
            The memory usage in megabytes (MB).
        """
        try:
            pid = os.getpid()
            process = psutil.Process(pid)
            # rss is the Resident Set Size, which is the non-swapped physical memory a process has used.
            memory_bytes = process.memory_info().rss
            return memory_bytes / (1024 * 1024)
        except Exception as e:
            self._logger.warning(f"[{self._actor_id_str}] Could not retrieve process memory usage: {e}")
            return 0.0

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
              - 'memory_mb' (float): The total memory usage of the current actor process (RSS) in megabytes (MB).
        """
        # If the actor is not running, return the last known stats to ensure this
        # call is non-blocking during shutdown.
        if not self._running:
            stats_copy = self.stats.copy()
            stats_copy["active_processing"] = False  # It's not active if not running
            stats_copy["memory_mb"] = self._get_memory_usage_mb()
            return stats_copy

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
            "memory_mb": self._get_memory_usage_mb(),
        }

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
        # Prevent starting if already running
        if self._running:
            self._logger.warning(f"{self._actor_id_str}: Start called but actor is already running.")
            return False

        self._logger.debug(f"{self._actor_id_str}: Starting actor...")
        # --- Initialize Actor State ---
        self._running = True
        self._shutting_down = False  # Reset shutdown flag on start
        self._shutdown_signal_complete = False
        self.start_time = time.time()

        # --- Reset Statistics ---
        self._last_stats_time = self.start_time
        self._last_processed_count = 0

        # --- Start Background Processing Thread ---
        self._logger.debug(f"{self._actor_id_str}: Creating and starting processing thread.")
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=False,
        )
        self._processing_thread.start()

        self._logger.debug(f"{self._actor_id_str}: Actor started successfully.")

        return True

    @ray.method(num_returns=0)
    def stop(self) -> None:
        """Stops the actor's processing loop by setting the running flag to False."""
        self._logger.debug(f"[{self._actor_id_str}] Stop signal received. Initiating graceful shutdown.")
        self._running = False

    def is_shutdown_complete(self) -> bool:
        """
        Checks if the actor's processing loop has finished and signaled completion.
        Raises RayActorError if the actor process has terminated.
        """
        return self._shutdown_signal_complete

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
        self._logger.debug(f"{self._actor_id_str}: Setting input queue.")
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
        self._logger.debug(f"{self._actor_id_str}: Setting output queue.")
        self._output_queue = queue_handle
        return True
