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

# --- Constants ---
CLEANUP_INTERVAL_SECONDS = 15.0
PENDING_SHUTDOWN_TIMEOUT_SECONDS = 60.0 * 60
PENDING_CHECK_ACTOR_METHOD_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


class StageInfo:
    def __init__(
        self,
        name,
        callable,
        config,
        is_source=False,
        is_sink=False,
        min_replicas=0,
        max_replicas=1,
        pending_shutdown=False,
    ):
        self.name = name
        self.callable = callable
        self.config = config
        self.is_source = is_source
        self.is_sink = is_sink
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.pending_shutdown = pending_shutdown


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
        self._pending_removal_actors: Dict[str, Set[Tuple[Any, str, float, ray.ObjectRef]]] = defaultdict(set)

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
        """Ensure cleanup thread is stopped and internal actor references are released."""
        logger.debug("PipelineTopology destructor called. Cleaning up thread and actor references.")

        # Stop the background cleanup thread
        try:
            self._stop_cleanup_thread()
        except Exception as e:
            logger.warning(f"Error stopping cleanup thread during __del__: {e}")

        # Clear references to actor handles and shutdown futures
        try:
            self._stage_actors.clear()
            self._edge_queues.clear()
            self._scaling_state.clear()
            self._stage_memory_overhead.clear()
            self._pending_removal_actors.clear()
            self._stages.clear()
            self._connections.clear()
        except Exception as e:
            logger.warning(f"Error clearing internal state during __del__: {e}")

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

    def register_actors_pending_removal(self, registration_info: Dict[str, List[Tuple[Any, ray.ObjectRef]]]) -> None:
        """
        Registers actor handles that have been told to stop, along with their shutdown futures.
        The topology's background thread will monitor these futures for completion.

        Parameters
        ----------
        registration_info : Dict[str, List[Tuple[Any, ObjectRef]]]
            Dictionary mapping stage names to a list of (actor_handle, shutdown_future) tuples.
        """
        added_count = 0
        time_registered = time.time()
        stages_updated = set()

        with self._lock:
            all_known_stages = {s.name for s in self._stages}

            for stage_name, actor_list in registration_info.items():
                if stage_name not in all_known_stages:
                    logger.warning(
                        f"[TopologyRegister] Received pending removal registration for unknown stage "
                        f"'{stage_name}'. Skipping."
                    )
                    continue

                stage_pending_set = self._pending_removal_actors[stage_name]

                for actor_handle, shutdown_future in actor_list:
                    if not actor_handle or not shutdown_future:
                        logger.warning(
                            f"[TopologyRegister-{stage_name}] "
                            f"Received invalid (actor, future) in registration list. Skipping."
                        )
                        continue

                    actor_id_str = str(actor_handle)
                    actor_tuple = (actor_handle, actor_id_str, time_registered, shutdown_future)

                    if actor_tuple not in stage_pending_set:
                        stage_pending_set.add(actor_tuple)
                        added_count += 1
                        logger.debug(
                            f"[TopologyRegister-{stage_name}] "
                            f"Registered actor '{actor_id_str}' pending shutdown monitoring."
                        )
                    else:
                        logger.debug(
                            f"[TopologyRegister-{stage_name}] "
                            f"Actor '{actor_id_str}' already registered pending removal."
                        )

                if actor_list:
                    self._scaling_state[stage_name] = "Scaling Down Pending"
                    stages_updated.add(stage_name)

        if added_count > 0:
            logger.debug(
                f"[TopologyRegister] Registered {added_count} "
                f"actors across {len(stages_updated)} stages pending removal."
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

    @staticmethod
    def _delayed_actor_release(self, actor_handle_to_release: Any, actor_id_str: str, delay_seconds: int = 60):
        """
        Holds a reference to an actor handle for a specified delay, then releases it.
        This function is intended to be run in a daemon thread.

        Note: this is a bit of a hack
        """
        logger.debug(f"[DelayedRelease-{actor_id_str}] Thread started. Holding actor reference for {delay_seconds}s.")
        # The actor_handle_to_release is kept in scope by being a parameter to this function,
        # and this function's frame existing for delay_seconds.
        time.sleep(delay_seconds)
        logger.info(
            f"[DelayedRelease-{actor_id_str}] Delay complete. Releasing reference. Actor should now be GC'd by Ray "
            f"if this was the last ref."
        )
        # When this function exits, actor_handle_to_release goes out of scope, dropping the reference.

    def _cleanup_loop(self) -> None:
        """
        Background thread for periodically checking shutdown status of actors pending removal.

        Actors are removed from the topology once their shutdown futures complete or they time out.
        """
        logger.info("[TopologyCleanupLoop] Cleanup thread started.")

        while self._cleanup_thread_running:
            cycle_start_time = time.time()
            actors_removed_this_cycle = 0
            processed_actor_ids_this_cycle = set()
            actors_to_remove_from_pending: Dict[str, List[Tuple[Any, str, float, ray.ObjectRef]]] = defaultdict(list)
            stages_potentially_idle: Set[str] = set()

            try:
                with self._lock:
                    if not self._cleanup_thread_running:
                        logger.debug(
                            "[TopologyCleanupLoop] " "Stop signal received after lock acquisition. Exiting loop."
                        )
                        break

                    for stage_name in list(self._pending_removal_actors.keys()):
                        pending_set = self._pending_removal_actors[stage_name]
                        if not pending_set:
                            continue

                        pending_set_copy = pending_set.copy()

                        for actor_tuple in pending_set_copy:
                            actor_handle, actor_id_str, time_registered, shutdown_future = actor_tuple

                            if actor_id_str in processed_actor_ids_this_cycle:
                                continue

                            remove_from_topology = False
                            mark_for_pending_removal = False
                            actor_status = "PENDING"

                            # 1. Check for overall shutdown timeout
                            if time.time() - time_registered > PENDING_SHUTDOWN_TIMEOUT_SECONDS:
                                logger.warning(
                                    f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                    f"timed out after {PENDING_SHUTDOWN_TIMEOUT_SECONDS}s. Forcing removal."
                                )
                                remove_from_topology = True
                                mark_for_pending_removal = True
                                actor_status = "TIMEOUT"

                            # 2. Otherwise, check if shutdown future completed
                            if not remove_from_topology:
                                try:
                                    ready, _ = ray.wait([shutdown_future], timeout=PENDING_CHECK_ACTOR_METHOD_TIMEOUT)
                                    if ready:
                                        logger.debug(
                                            f"[TopologyCleanupLoop-{stage_name}] "
                                            f"Actor '{actor_id_str}' shutdown future completed. Marking for removal."
                                        )
                                        remove_from_topology = True
                                        mark_for_pending_removal = True
                                        actor_status = "COMPLETED"
                                    else:
                                        logger.debug(
                                            f"[TopologyCleanupLoop-{stage_name}] "
                                            f"Actor '{actor_id_str}' shutdown future still pending."
                                        )
                                        actor_status = "PENDING"
                                except Exception as e:
                                    logger.error(
                                        f"[TopologyCleanupLoop-{stage_name}] "
                                        f"Error checking shutdown future for actor '{actor_id_str}': {e}",
                                        exc_info=False,
                                    )
                                    actor_status = "ERROR"

                            # 3. Perform removal actions
                            if remove_from_topology:
                                logger.debug(
                                    f"[TopologyCleanupLoop-{stage_name}] Removing actor '{actor_id_str}' "
                                    f"from topology (Reason: {actor_status})."
                                )
                                removed_list = self.remove_actors_from_stage(stage_name, [actor_handle])
                                if removed_list:
                                    actors_removed_this_cycle += 1
                                else:
                                    logger.debug(
                                        f"[TopologyCleanupLoop-{stage_name}] Actor '{actor_id_str}' "
                                        f"was already removed from main list."
                                    )

                            if mark_for_pending_removal:
                                actors_to_remove_from_pending[stage_name].append(actor_tuple)
                                processed_actor_ids_this_cycle.add(actor_id_str)
                                stages_potentially_idle.add(stage_name)

                    # --- Update pending lists ---
                    for stage_to_update, removal_list in actors_to_remove_from_pending.items():
                        if stage_to_update in self._pending_removal_actors:
                            current_pending_set = self._pending_removal_actors[stage_to_update]
                            for removal_tuple in removal_list:  # removal_list contains actor_tuples
                                # Extract actor_handle and actor_id_str from the tuple being removed
                                actor_handle_to_delay, actor_id_str_to_delay, _, _ = removal_tuple

                                if current_pending_set.discard(
                                    removal_tuple
                                ):  # If discard was successful (element was present)
                                    logger.debug(
                                        f"[TopologyCleanupLoop-{stage_to_update}] Actor tuple for "
                                        f"'{actor_id_str_to_delay}' discarded from pending set."
                                    )
                                    try:
                                        # This is a bit of a hack. For some reason Ray likes to cause exceptions on
                                        # the actor when we let it auto GCS just after pushing to the output queue, and
                                        # mysteriously lose control messages.
                                        # This lets the shutdown future complete, but leaves the actor to be killed off
                                        # by ray.actor_exit()
                                        delay_thread = threading.Thread(
                                            target=self._delayed_actor_release,
                                            args=(actor_handle_to_delay, actor_id_str_to_delay, 60),  # 60s delay
                                            daemon=True,
                                        )
                                        delay_thread.start()
                                        logger.debug(
                                            f"[TopologyCleanupLoop-{stage_to_update}] Started delayed release thread "
                                            f"for '{actor_id_str_to_delay}'."
                                        )
                                    except Exception as e_thread:
                                        logger.error(
                                            f"[TopologyCleanupLoop-{stage_to_update}] Failed to start delayed release "
                                            f"thread for '{actor_id_str_to_delay}': {e_thread}"
                                        )

                            # After processing all removals for this stage's list, check if the set is empty
                            if not self._pending_removal_actors[stage_to_update]:
                                logger.debug(
                                    f"[TopologyCleanupLoop-{stage_to_update}] Pending set empty. Deleting key."
                                )
                                del self._pending_removal_actors[stage_to_update]

                    # --- Update stage scaling states if pending list is empty ---
                    stages_with_empty_pending = []
                    stages_with_empty_pending = []
                    for stage_to_check in stages_potentially_idle:
                        if stage_to_check not in self._pending_removal_actors:
                            stages_with_empty_pending.append(stage_to_check)
                            if self._scaling_state.get(stage_to_check) == "Scaling Down Pending":
                                logger.debug(  # Your original log level
                                    f"[TopologyCleanupLoop-{stage_to_check}] All pending actors cleared. "
                                    f"Setting scaling state to Idle."
                                )
                                self._scaling_state[stage_to_check] = "Idle"

                    # --- Log cycle summary ---
                    cycle_duration = time.time() - cycle_start_time
                    if actors_removed_this_cycle > 0:
                        logger.debug(
                            f"[TopologyCleanupLoop] Cleanup cycle finished in {cycle_duration:.3f}s. "
                            f"Removed {actors_removed_this_cycle} actors."
                        )
                    else:
                        logger.debug(
                            f"[TopologyCleanupLoop] Cleanup cycle finished in {cycle_duration:.3f}s. "
                            f"No actors removed."
                        )

            except Exception as e:
                logger.error(f"[TopologyCleanupLoop] Unhandled error in cleanup loop iteration: " f"{e}", exc_info=True)

            # --- Wait until next cycle ---
            woken_by_stop = self._stop_event.wait(timeout=CLEANUP_INTERVAL_SECONDS)
            if woken_by_stop:
                logger.info("[TopologyCleanupLoop] Stop event received during sleep. Exiting loop.")
                break

        logger.info("[TopologyCleanupLoop] Cleanup thread finished.")

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

            logger.debug("Cleared runtime state (actors, queues, scaling state, flushing flag).")

    # --- Accessor Methods (Read Operations - Use Lock, Return Copies) ---

    def get_stages_info(self) -> List[StageInfo]:
        """Returns a copy of stage info with pending_shutdown flags updated."""
        with self._lock:
            updated_stages = []
            for stage in self._stages:
                pending_shutdown = bool(self._pending_removal_actors.get(stage.name))
                # Make a shallow copy with updated pending_shutdown
                stage_copy = StageInfo(
                    name=stage.name,
                    callable=stage.callable,
                    config=stage.config,
                    is_source=stage.is_source,
                    is_sink=stage.is_sink,
                    min_replicas=stage.min_replicas,
                    max_replicas=stage.max_replicas,
                    pending_shutdown=pending_shutdown,
                )
                updated_stages.append(stage_copy)
            return updated_stages

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
