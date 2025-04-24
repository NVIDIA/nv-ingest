# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import logging
import contextlib
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Iterator, Set

import ray
from ray.exceptions import RayActorError, GetTimeoutError

# --- Constants ---
CLEANUP_INTERVAL_SECONDS = 15.0
PENDING_SHUTDOWN_TIMEOUT_SECONDS = 300.0
PENDING_CHECK_ACTOR_METHOD_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


class StageInfo:
    def __init__(self, name, callable, config, is_source=False, is_sink=False, min_replicas=0, max_replicas=1):
        self.name = name
        self.callable = callable
        self.config = config
        self.is_source = is_source
        self.is_sink = is_sink
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas


class PipelineTopology:
    """
    Holds the structural definition and runtime state of the pipeline.

    Encapsulates stages, connections, actors, queues, and associated state
    with thread-safe access via internal locking.
    """

    def __init__(self):
        # --- Definition ---
        self._stages: List[StageInfo] = []
        self._connections: Dict[str, List[Tuple[str, int]]] = {}

        # --- Runtime State ---
        self._stage_actors: Dict[str, List[Any]] = {}
        self._edge_queues: Dict[str, Tuple[Any, int]] = {}  # Map: q_name -> (QueueHandle, Capacity)
        self._scaling_state: Dict[str, str] = {}  # Map: stage_name -> "Idle" | "Scaling Up" | "Scaling Down" | "Error"
        self._stage_memory_overhead: Dict[str, float] = {}  # Populated during build/config
        self._pending_removal_actors: Dict[str, Set[Tuple[Any, str, float, str]]] = defaultdict(set)

        # --- Operational State ---
        self._is_flushing: bool = False

        # --- Synchronization & Threading ---
        self._lock: threading.Lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_thread_running: bool = False
        self._stop_event: threading.Event = threading.Event()  # For interruptible sleep

        logger.debug("PipelineTopology initialized.")
        self._start_cleanup_thread()  # Start background cleanup on init

    def __del__(self):
        """Ensure cleanup thread is stopped when topology object is destroyed."""
        logger.debug("PipelineTopology destructor called, ensuring cleanup thread is stopped.")
        self._stop_cleanup_thread()

    # --- Lock Context Manager ---
    @contextlib.contextmanager
    def lock_context(self) -> Iterator["PipelineTopology"]:
        """Provides safe access to the topology under lock for complex operations."""
        with self._lock:
            yield self

    # --- Mutator Methods (Write Operations - Use Lock) ---

    def add_stage(self, stage_info: StageInfo) -> None:
        """Adds a stage definition."""
        with self._lock:
            # Prevent duplicate stage names?
            if any(s.name == stage_info.name for s in self._stages):
                logger.error(f"Attempted to add duplicate stage name: {stage_info.name}")
                raise ValueError(f"Stage name '{stage_info.name}' already exists.")
            self._stages.append(stage_info)
            logger.debug(f"Added stage definition: {stage_info.name}")

    def add_connection(self, from_stage: str, to_stage: str, queue_size: int) -> None:
        """Adds a connection definition between two stages."""
        with self._lock:
            # Basic validation (more can be added in Pipeline class)
            stage_names = {s.name for s in self._stages}
            if from_stage not in stage_names:
                raise ValueError(f"Source stage '{from_stage}' for connection not found.")
            if to_stage not in stage_names:
                raise ValueError(f"Destination stage '{to_stage}' for connection not found.")

            self._connections.setdefault(from_stage, []).append((to_stage, queue_size))
            logger.debug(f"Added connection definition: {from_stage} -> {to_stage} (q_size={queue_size})")

    def set_actors_for_stage(self, stage_name: str, actors: List[Any]) -> None:
        """Sets the list of actors for a given stage, resetting scaling state."""
        with self._lock:
            if stage_name not in {s.name for s in self._stages}:
                logger.warning(f"Attempted to set actors for unknown stage: {stage_name}")
                return  # Or raise error?
            self._stage_actors[stage_name] = actors
            self._scaling_state[stage_name] = "Idle"  # Initialize/reset state
            logger.debug(f"Set {len(actors)} actors for stage '{stage_name}'. State set to Idle.")

    def add_actor_to_stage(self, stage_name: str, actor: Any) -> None:
        """Adds a single actor to a stage's list."""
        with self._lock:
            if stage_name not in self._stage_actors:
                # This might happen if stage has 0 min_replicas and is scaled up first time
                self._stage_actors[stage_name] = []
                self._scaling_state[stage_name] = "Idle"  # Ensure state exists
                logger.debug(f"Initialized actor list for stage '{stage_name}' during add.")
            self._stage_actors[stage_name].append(actor)
            logger.debug(f"Added actor to stage '{stage_name}'. New count: {len(self._stage_actors[stage_name])}")

    def remove_actors_from_stage(self, stage_name: str, actors_to_remove: List[Any]) -> List[Any]:
        """
        Removes specific actors from a stage's list immediately.
        Called by the cleanup thread or potentially for forced removal.
        """
        removed = []
        # Assumes lock is already held by caller (e.g., cleanup thread or lock_context)
        if stage_name not in self._stage_actors:
            logger.warning(
                f"[Topology-InternalRemove] Attempted to remove actors from non-existent stage entry: {stage_name}"
            )
            return []
        current_actors = self._stage_actors.get(stage_name, [])

        # Create sets for efficient lookup
        current_actor_set = set(current_actors)
        to_remove_set = set(actors_to_remove)

        # Actors remaining are those in current set but not in removal set
        actors_remaining = list(current_actor_set - to_remove_set)
        # Actors actually removed are the intersection
        actors_actually_removed = list(current_actor_set.intersection(to_remove_set))

        if actors_actually_removed:
            self._stage_actors[stage_name] = actors_remaining
            removed = actors_actually_removed
            logger.debug(
                f"[Topology-InternalRemove] Removed {len(removed)} actors from stage '{stage_name}'. "
                f"Remaining: {len(actors_remaining)}"
            )
        elif to_remove_set:
            # This might happen if called twice for the same actor
            logger.debug(f"[Topology-InternalRemove] No actors matching removal list found in stage '{stage_name}'.")

        return removed

    def register_actors_pending_removal(self, registration_info: Dict[str, List[Any]]) -> None:
        """
        Registers actor handles that have been told to stop. The topology's background
        thread will monitor these for completion.

        Parameters
        ----------
        registration_info : Dict[str, List[Any]]
            Dictionary mapping stage names to a list of actor handles.
        """
        added_count = 0
        time_registered = time.time()
        stages_updated = set()

        # Acquire lock to safely modify pending list and scaling state
        with self._lock:
            all_known_stages = {s.name for s in self._stages}  # Get current stage names under lock

            for stage_name, actor_list in registration_info.items():
                # Validate stage name
                if stage_name not in all_known_stages:
                    logger.warning(
                        f"[TopologyRegister] Received pending removal registration for unknown stage "
                        f"'{stage_name}'. Skipping."
                    )
                    continue

                # Get or initialize the pending set for this stage
                stage_pending_set = self._pending_removal_actors[stage_name]

                for actor_handle in actor_list:
                    if not actor_handle:  # Skip None or invalid handles
                        logger.warning(
                            f"[TopologyRegister-{stage_name}] Received invalid handle in registration list. Skipping."
                        )
                        continue

                    actor_id_str = str(actor_handle)  # Use string representation for logging/lookup
                    # Store tuple: (handle, id_string, timestamp)
                    actor_tuple = (actor_handle, actor_id_str, time_registered)

                    # Add to set (sets handle duplicates automatically)
                    if actor_tuple not in stage_pending_set:
                        stage_pending_set.add(actor_tuple)
                        added_count += 1
                        logger.debug(
                            f"[TopologyRegister-{stage_name}] Registered actor '{actor_id_str}' pending removal."
                        )
                    else:
                        # This case is less likely now that stop() confirms before registration
                        logger.debug(
                            f"[TopologyRegister-{stage_name}] Actor '{actor_id_str}' "
                            f"was already registered pending removal. Ignoring duplicate."
                        )

                # Mark stage as having pending removals if actors were added
                if actor_list:
                    self._scaling_state[stage_name] = "Scaling Down Pending"
                    stages_updated.add(stage_name)

        if added_count > 0:
            logger.info(
                f"[TopologyRegister] Registered {added_count} actors across "
                f"{len(stages_updated)} stages pending removal."
            )
        elif registration_info:
            logger.debug("[TopologyRegister] No new actors registered pending removal (likely duplicates).")

    def _start_cleanup_thread(self) -> None:
        """Starts the background thread for cleaning up terminated actors."""
        with self._lock:  # Protect thread state modification
            if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
                logger.warning("[TopologyCleanup] Cleanup thread already started.")
                return

            logger.info("[TopologyCleanup] Starting background cleanup thread...")
            self._cleanup_thread_running = True
            self._stop_event.clear()  # Ensure event is not set initially
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,  # Allows program exit even if this thread hangs (though join tries)
                name="TopologyActorCleanup",
            )
            self._cleanup_thread.start()

    def _stop_cleanup_thread(self) -> None:
        """Signals the background cleanup thread to stop and waits for it."""
        if not self._cleanup_thread_running or self._cleanup_thread is None:
            logger.debug("[TopologyCleanup] Cleanup thread not running or already stopped.")
            return

        with self._lock:  # Protect thread state read/write
            if not self._cleanup_thread_running or self._cleanup_thread is None:
                return  # Double check inside lock
            logger.info("[TopologyCleanup] Stopping background cleanup thread...")
            self._cleanup_thread_running = False
            self._stop_event.set()  # Signal the loop to wake up and exit

        # Wait for the thread to finish outside the lock
        join_timeout = CLEANUP_INTERVAL_SECONDS + 5.0  # Give it time to finish last cycle
        self._cleanup_thread.join(timeout=join_timeout)

        if self._cleanup_thread.is_alive():
            logger.warning(f"[TopologyCleanup] Cleanup thread did not exit gracefully after {join_timeout}s.")
        else:
            logger.info("[TopologyCleanup] Cleanup thread stopped and joined.")
        self._cleanup_thread = None  # Clear thread object

    def _cleanup_loop(self) -> None:
        """
        The main loop for the background cleanup thread. Periodically checks actors
        pending removal, updates topology, and removes them from the pending list.
        """
        logger.info("[TopologyCleanupLoop] Cleanup thread started.")

        while self._cleanup_thread_running:
            cycle_start_time = time.time()
            actors_removed_this_cycle = 0
            processed_actor_ids_this_cycle = set()
            # Store actors to remove from pending list {stage: [tuple(handle, id, time)]}
            actors_to_remove_from_pending: Dict[str, List[Tuple[Any, str, float]]] = defaultdict(list)
            # Track stages that had removals to check if their pending list is now empty
            stages_potentially_idle: Set[str] = set()

            try:
                # --- Acquire Lock for read/write operations ---
                with self._lock:
                    # Double-check running flag after acquiring lock
                    if not self._cleanup_thread_running:
                        logger.debug("[TopologyCleanupLoop] Stop signal received after lock acquisition. Exiting loop.")
                        break

                    # Iterate over stages that have pending actors
                    # Use list() to copy keys in case dict changes during iteration (less likely here due to lock)
                    for stage_name in list(self._pending_removal_actors.keys()):
                        pending_set = self._pending_removal_actors[stage_name]
                        if not pending_set:
                            continue  # Skip if somehow empty

                        # Iterate over a copy of the set to allow safe removal marking
                        pending_set_copy = pending_set.copy()

                        for actor_tuple in pending_set_copy:
                            # Tuple contains: (handle, id_string, registration_timestamp)
                            actor_handle, actor_id_str, time_registered = actor_tuple

                            # Avoid reprocessing an actor multiple times within one cycle
                            if actor_id_str in processed_actor_ids_this_cycle:
                                continue

                            remove_from_topology = False
                            mark_for_pending_removal = False
                            actor_status = "PENDING"  # PENDING, COMPLETED, ACTOR_GONE, TIMEOUT, UNRESPONSIVE, ERROR

                            # 1. Check for Overall Timeout first
                            if time.time() - time_registered > PENDING_SHUTDOWN_TIMEOUT_SECONDS:
                                logger.warning(
                                    f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                    f"timed out waiting for shutdown "
                                    f"({PENDING_SHUTDOWN_TIMEOUT_SECONDS}s). Forcing removal."
                                )
                                remove_from_topology = True
                                mark_for_pending_removal = True
                                actor_status = "TIMEOUT"

                            # 2. If not timed out, check actor status via is_shutdown_complete
                            if not remove_from_topology:
                                try:
                                    # Call the remote method on the actor handle
                                    check_ref = actor_handle.is_shutdown_complete.remote()
                                    # Wait for the boolean result with a specific timeout
                                    is_complete = ray.get(check_ref, timeout=PENDING_CHECK_ACTOR_METHOD_TIMEOUT)

                                    if is_complete:
                                        # Actor's loop finished, signal flag is True
                                        logger.info(
                                            f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                            f"reported shutdown complete (flag is True). Marking for removal."
                                        )
                                        remove_from_topology = True
                                        mark_for_pending_removal = True
                                        actor_status = "COMPLETED"
                                    else:
                                        # Actor's loop still running, flag is False
                                        logger.debug(
                                            f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                            f"reported still running (flag is False). Will check again."
                                        )
                                        actor_status = "PENDING"

                                except RayActorError:
                                    # The actor handle is invalid, meaning the process has terminated.
                                    logger.warning(
                                        f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                        f"handle is invalid (RayActorError raised). "
                                        f"Assuming actor terminated. Marking for removal."
                                    )
                                    remove_from_topology = True
                                    mark_for_pending_removal = True
                                    actor_status = "ACTOR_GONE"
                                except (GetTimeoutError, TimeoutError):
                                    # The call to is_shutdown_complete timed out
                                    logger.warning(
                                        f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                        f"status check timed out after {PENDING_CHECK_ACTOR_METHOD_TIMEOUT}s. "
                                        f"Assuming unresponsive, will rely on overall timeout."
                                    )
                                    actor_status = "UNRESPONSIVE"
                                except Exception as e:
                                    # Catch other potential errors during the check
                                    logger.error(
                                        f"[TopologyCleanupLoop-{stage_name}] Error checking status for actor "
                                        f"'{actor_id_str}': {e}",
                                        exc_info=False,
                                    )
                                    actor_status = "ERROR"  # Rely on overall timeout

                            # 3. Perform Actions based on findings (still inside the lock)
                            if remove_from_topology:
                                logger.info(
                                    f"[TopologyCleanupLoop-{stage_name}] Removing actor '{actor_id_str}' "
                                    f"from main topology list (Reason: {actor_status})."
                                )
                                # Use internal method which assumes lock is held
                                # Pass the specific handle to remove
                                removed_list = self.remove_actors_from_stage(stage_name, [actor_handle])
                                if removed_list:
                                    actors_removed_this_cycle += 1
                                else:
                                    # This might happen if removal was attempted twice rapidly
                                    logger.debug(
                                        f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                        f"was apparently already removed from main list."
                                    )

                            # 4. Mark for removal from the pending list if needed
                            if mark_for_pending_removal:
                                logger.debug(
                                    f"[TopologyCleanupLoop-{stage_name}] Marking actor '{actor_id_str}' "
                                    f"for removal from pending list."
                                )
                                actors_to_remove_from_pending[stage_name].append(actor_tuple)
                                processed_actor_ids_this_cycle.add(actor_id_str)  # Use ID string for tracking
                                stages_potentially_idle.add(stage_name)  # Track stage

                    # --- Update the pending removal data structure (still inside lock) ---
                    for stage_to_update, actors_list_to_remove in actors_to_remove_from_pending.items():
                        # Check if stage still exists in pending dict (might have been removed if empty)
                        if stage_to_update in self._pending_removal_actors:
                            current_pending_set = self._pending_removal_actors[stage_to_update]
                            for removal_tuple in actors_list_to_remove:
                                current_pending_set.discard(removal_tuple)  # Safely remove tuple

                    # --- Check stages whose pending lists might now be empty (still inside lock) ---
                    stages_with_empty_pending = []
                    for stage_to_check in stages_potentially_idle:
                        # Check if the key exists AND the set is empty
                        if (
                            stage_to_check in self._pending_removal_actors
                            and not self._pending_removal_actors[stage_to_check]
                        ):
                            stages_with_empty_pending.append(stage_to_check)
                            # Clean up the entry from the main dictionary
                            del self._pending_removal_actors[stage_to_check]
                            # If state was pending, set back to Idle
                            if self._scaling_state.get(stage_to_check) == "Scaling Down Pending":
                                logger.info(
                                    f"[TopologyCleanupLoop-{stage_to_check}] All pending actors cleared for stage. "
                                    f"Setting scaling state to Idle."
                                )
                                self._scaling_state[stage_to_check] = "Idle"

                # --- End of lock block ---

                # --- Logging for the cycle ---
                cycle_duration = time.time() - cycle_start_time
                if actors_removed_this_cycle > 0:
                    logger.info(
                        f"[TopologyCleanupLoop] Cleanup cycle finished in {cycle_duration:.3f}s. "
                        f"Removed {actors_removed_this_cycle} actors from topology."
                    )
                else:
                    logger.debug(
                        f"[TopologyCleanupLoop] Cleanup cycle finished in {cycle_duration:.3f}s. "
                        f"No actors removed from topology."
                    )

            except Exception as e:
                # Catch broad exceptions in the loop itself to prevent thread death
                logger.error("[TopologyCleanupLoop] Unhandled error in cleanup loop iteration: " f"{e}", exc_info=True)

            # --- Wait for the next interval or stop signal ---
            # Use event.wait for interruptible sleep outside the main try/except
            logger.debug(f"[TopologyCleanupLoop] Sleeping for {CLEANUP_INTERVAL_SECONDS}s...")
            woken_by_stop = self._stop_event.wait(timeout=CLEANUP_INTERVAL_SECONDS)
            if woken_by_stop:
                logger.info("[TopologyCleanupLoop] Stop event received during sleep. Exiting.")
                break  # Exit the while loop cleanly

        logger.info("[TopologyCleanupLoop] Cleanup loop thread finished.")

    def set_edge_queues(self, queues: Dict[str, Tuple[Any, int]]) -> None:
        """Sets the dictionary of edge queues."""
        with self._lock:
            self._edge_queues = queues
            logger.debug(f"Set {len(queues)} edge queues.")

    def update_scaling_state(self, stage_name: str, state: str) -> None:
        """Updates the scaling state for a stage."""
        with self._lock:
            # Add validation for state values?
            valid_states = {"Idle", "Scaling Up", "Scaling Down", "Error"}
            if state not in valid_states:
                logger.error(f"Invalid scaling state '{state}' for stage '{stage_name}'. Ignoring.")
                return
            if stage_name not in {s.name for s in self._stages}:
                logger.warning(f"Attempted to set scaling state for unknown stage: {stage_name}")
                return
            self._scaling_state[stage_name] = state
            logger.debug(f"Updated scaling state for '{stage_name}' to '{state}'.")

    def set_flushing(self, is_flushing: bool) -> None:
        """Sets the pipeline flushing state."""
        with self._lock:
            self._is_flushing = is_flushing
            logger.debug(f"Pipeline flushing state set to: {is_flushing}")

    def set_stage_memory_overhead(self, overheads: Dict[str, float]) -> None:
        """Sets the estimated memory overhead for stages."""
        with self._lock:
            self._stage_memory_overhead = overheads
            logger.debug(f"Set memory overheads for {len(overheads)} stages.")

    def clear_runtime_state(self) -> None:
        """Clears actors, queues, and scaling state. Keeps definitions."""
        with self._lock:
            self._stage_actors.clear()
            self._edge_queues.clear()
            self._scaling_state.clear()
            self._is_flushing = False  # Reset flushing state too
            logger.info("Cleared runtime state (actors, queues, scaling state, flushing flag).")

    # --- Accessor Methods (Read Operations - Use Lock, Return Copies) ---

    def get_stages_info(self) -> List[StageInfo]:
        """Returns a copy of the list of stage information."""
        with self._lock:
            return self._stages[:]

    def get_stage_info(self, stage_name: str) -> Optional[StageInfo]:
        """Returns the StageInfo for a specific stage, or None if not found."""
        with self._lock:
            for stage in self._stages:
                if stage.name == stage_name:
                    return stage
            return None

    def get_connections(self) -> Dict[str, List[Tuple[str, int]]]:
        """Returns a shallow copy of the connections dictionary."""
        with self._lock:
            # Shallow copy is usually sufficient here as tuples are immutable
            return self._connections.copy()

    def get_stage_actors(self) -> Dict[str, List[Any]]:
        """Returns a copy of the stage actors dictionary (with copies of actor lists)."""
        with self._lock:
            return {name: list(actors) for name, actors in self._stage_actors.items()}

    def get_actor_count(self, stage_name: str) -> int:
        """Returns the number of actors for a specific stage."""
        with self._lock:
            return len(self._stage_actors.get(stage_name, []))

    def get_edge_queues(self) -> Dict[str, Tuple[Any, int]]:
        """Returns a shallow copy of the edge queues dictionary."""
        with self._lock:
            return self._edge_queues.copy()

    def get_scaling_state(self) -> Dict[str, str]:
        """Returns a copy of the scaling state dictionary."""
        with self._lock:
            return self._scaling_state.copy()

    def get_is_flushing(self) -> bool:
        """Returns the current flushing state."""
        with self._lock:
            return self._is_flushing

    def get_stage_memory_overhead(self) -> Dict[str, float]:
        """Returns a copy of the stage memory overhead dictionary."""
        with self._lock:
            return self._stage_memory_overhead.copy()
