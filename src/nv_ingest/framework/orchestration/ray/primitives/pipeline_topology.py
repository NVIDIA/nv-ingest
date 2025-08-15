# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import logging
import contextlib
import time
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
        self._actors_pending_removal: Set[Tuple[str, Any]] = set()

        # --- Operational State ---
        self._is_flushing: bool = False

        # --- Synchronization & Threading ---
        self._lock: threading.Lock = threading.Lock()
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = None

    def __del__(self):
        """Ensure cleanup thread is stopped and internal actor references are released."""
        logger.debug("PipelineTopology destructor called. Cleaning up thread and actor references.")

        # Stop the background cleanup thread
        try:
            self.stop_cleanup_thread()
        except Exception as e:
            logger.warning(f"Error stopping cleanup thread during __del__: {e}")

        # Clear references to actor handles and shutdown futures
        try:
            self._stage_actors.clear()
            self._edge_queues.clear()
            self._scaling_state.clear()
            self._stage_memory_overhead.clear()
            self._actors_pending_removal.clear()
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

    def mark_actor_for_removal(self, stage_name: str, actor: Any) -> None:
        """Marks an actor as pending removal, to be cleaned up by the background thread."""
        with self._lock:
            self._actors_pending_removal.add((stage_name, actor))
            logger.debug(f"Marked actor {actor} from stage {stage_name} for removal.")

    def start_cleanup_thread(self, interval: int = 5) -> None:
        """Starts the background thread for periodic cleanup tasks."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, args=(interval,), daemon=True)
            self._cleanup_thread.start()
            logger.debug("Topology cleanup thread started.")

    def stop_cleanup_thread(self) -> None:
        """Stops the background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.debug("Topology cleanup thread stopped.")

    def _cleanup_loop(self, interval: int) -> None:
        """Periodically checks for and removes actors that have completed shutdown."""
        while not self._stop_cleanup.is_set():
            actors_to_remove_finally = []
            if not self._actors_pending_removal:
                time.sleep(interval)
                continue

            # Check the status of actors pending removal
            # Create a copy for safe iteration, as the set might be modified elsewhere
            pending_actors_copy = set()
            with self._lock:
                pending_actors_copy = set(self._actors_pending_removal)

            for stage_name, actor in pending_actors_copy:
                try:
                    if ray.get(actor.is_shutdown_complete.remote()):
                        actors_to_remove_finally.append((stage_name, actor))
                except ray.exceptions.RayActorError:
                    logger.warning(
                        f"Actor {actor} from stage {stage_name} is no longer available (RayActorError). "
                        f"Assuming it has shut down and marking for removal."
                    )
                    actors_to_remove_finally.append((stage_name, actor))
                except Exception as e:
                    logger.error(f"Error checking shutdown status for actor {actor}: {e}", exc_info=True)

            # Remove the fully shut-down actors from the topology
            if actors_to_remove_finally:
                with self._lock:
                    for stage_name, actor in actors_to_remove_finally:
                        if (stage_name, actor) in self._actors_pending_removal:
                            self._actors_pending_removal.remove((stage_name, actor))
                        if actor in self._stage_actors.get(stage_name, []):
                            self._stage_actors[stage_name].remove(actor)
                            logger.debug(f"Successfully removed actor {actor} from stage {stage_name} in topology.")

            time.sleep(interval)

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

    def get_all_actors(self) -> List[Any]:
        """Returns a list of all actors across all stages."""
        with self._lock:
            return [actor for actors in self._stage_actors.values() for actor in actors]

    def get_stages_info(self) -> List[StageInfo]:
        """Returns a copy of stage info with pending_shutdown flags updated."""
        with self._lock:
            updated_stages = []
            for stage in self._stages:
                pending_shutdown = bool(self._actors_pending_removal)
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
        """Returns a shallow copy of the connection dictionary."""
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
        """Returns a shallow copy of the edge queues' dictionary."""
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
