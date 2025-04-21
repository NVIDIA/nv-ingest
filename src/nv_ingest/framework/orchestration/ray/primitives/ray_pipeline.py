# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import psutil
import uuid
import ray
from ray.exceptions import GetTimeoutError
from ray.util.queue import Queue
from typing import Dict, Optional, List, Tuple, Any
from pydantic import BaseModel
import concurrent.futures
import logging
import time

from nv_ingest.framework.orchestration.ray.primitives.ray_stat_collector import RayStatsCollector
from nv_ingest.framework.orchestration.ray.util.pipeline.pid_controller import PIDController, ResourceConstraintManager
from nv_ingest.framework.orchestration.ray.util.system_tools.memory import estimate_actor_memory_overhead
from nv_ingest.framework.orchestration.ray.util.system_tools.visualizers import (
    GuiUtilizationDisplay,
    UtilizationDisplay,
)

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


class RayPipeline:
    """
    A structured pipeline supporting dynamic scaling and queue flushing.
    Uses PIDController and ResourceConstraintManager. Supports optional GUI display.
    Delegates statistics collection to RayStatsCollector.
    """

    def __init__(
        self,
        gui: bool = False,
        dynamic_memory_scaling: bool = False,
        dynamic_memory_threshold: float = 0.75,
        queue_flush_interval_seconds: int = 600,
        queue_flush_drain_timeout_seconds: int = 300,
        quiet_period_threshold: int = 0,
        pid_kp: float = 0.1,
        pid_ki: float = 0.001,
        pid_kd: float = 0.0,
        pid_target_queue_depth: int = 0,
        pid_penalty_factor: float = 0.1,
        pid_error_boost_factor: float = 1.5,
        pid_window_size: int = 10,
        rcm_estimated_edge_cost_mb: int = 5000,
        rcm_memory_safety_buffer_fraction: float = 0.15,
        # --- Stats Collector Config ---
        stats_collection_interval_seconds: float = 10.0,
        stats_actor_timeout_seconds: float = 5.0,
        stats_queue_timeout_seconds: float = 2.0,
    ) -> None:
        self.use_gui = gui

        # --- Core Pipeline Structure ---
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[Tuple[str, int]]] = {}
        self.stage_actors: Dict[str, List[Any]] = {}  # Map: stage_name -> List[ActorHandle]
        self.edge_queues: Dict[str, Tuple[Any, int]] = {}  # Map: q_name -> Tuple[QueueHandle, Capacity]

        # --- Structure Lock ---
        # Protects stages, stage_actors, edge_queues during modification and access
        self._structure_lock: threading.Lock = threading.Lock()

        # --- State ---
        self.scaling_state: Dict[str, str] = {}
        self.prev_global_memory_usage: Optional[int] = None

        # --- Build Time Config & State ---
        self.dynamic_memory_scaling = dynamic_memory_scaling
        self.dynamic_memory_threshold = dynamic_memory_threshold
        self.stage_memory_overhead: Dict[str, float] = {}

        # --- Background Threads (Keep monitoring/scaling threads if separate) ---
        self._queue_monitoring_thread: Optional[threading.Thread] = None  # Example
        self._scaling_thread: Optional[threading.Thread] = None  # Example
        self._display_instance: Optional[Any] = None
        self._monitoring = False
        self._scaling_monitoring = False

        # --- Queue Flushing ---
        self._is_flushing: bool = False
        self._last_queue_flush_time: float = time.time()
        self.queue_flush_interval_seconds = queue_flush_interval_seconds
        self.queue_flush_drain_timeout_seconds = queue_flush_drain_timeout_seconds
        self.quiet_period_threshold = quiet_period_threshold

        # --- Instantiate Autoscaling Controllers ---
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            stage_cost_estimates={},
            target_queue_depth=pid_target_queue_depth,
            window_size=pid_window_size,
            penalty_factor=pid_penalty_factor,
            error_boost_factor=pid_error_boost_factor,
        )
        logger.info("PIDController initialized...")
        try:
            total_system_memory_bytes = psutil.virtual_memory().total
            absolute_memory_threshold_mb = int(
                self.dynamic_memory_threshold * total_system_memory_bytes / (1024 * 1024)
            )
        except Exception as e:
            logger.error(f"Failed to get system memory: {e}. Using high limit.")
            absolute_memory_threshold_mb = 1_000_000
        self.constraint_manager = ResourceConstraintManager(
            max_replicas=1,
            memory_threshold=absolute_memory_threshold_mb,
            estimated_edge_cost_mb=rcm_estimated_edge_cost_mb,
            memory_safety_buffer_fraction=rcm_memory_safety_buffer_fraction,
        )  # Add real controller import/init
        logger.info("ResourceConstraintManager initialized...")

        # --- Instantiate Stats Collector ---
        # Store config values needed for staleness checks later
        self._stats_collection_interval_seconds = stats_collection_interval_seconds
        self.stats_collector = RayStatsCollector(
            pipeline_accessor=self,
            interval=stats_collection_interval_seconds,
            actor_timeout=stats_actor_timeout_seconds,
            queue_timeout=stats_queue_timeout_seconds,
        )

        logger.info(f"GUI Mode Requested: {self.use_gui}")
        logger.info("RayPipeline initialized.")

    # --- Accessor Methods for Stats Collector (and internal use) ---

    def get_stages_info(self) -> List[StageInfo]:
        """Returns a snapshot of the current stage information."""
        with self._structure_lock:
            return self.stages[:]  # Return a copy

    def get_stage_actors(self) -> Dict[str, List[Any]]:
        """Returns a snapshot of the current actors per stage."""
        with self._structure_lock:
            return {name: list(actors) for name, actors in self.stage_actors.items()}

    def get_edge_queues(self) -> Dict[str, Tuple[Any, int]]:
        """Returns a snapshot of the current edge queues."""
        with self._structure_lock:
            return self.edge_queues.copy()

    def _perform_dynamic_memory_scaling(self) -> None:
        """
        Estimates stage memory overhead and adjusts max_replicas if dynamic scaling is enabled.
        Updates self.stage_memory_overhead and potentially modifies self.stages[...].max_replicas.
        """
        return  # disable dynamic memory scaling for testing
        if not self.dynamic_memory_scaling:
            logger.info("Dynamic memory scaling disabled, skipping build-time adjustment.")
            return

        logger.info("Dynamic memory scaling enabled. Estimating per-stage memory overhead...")
        total_overhead = 0.0
        stage_overheads = {}  # Temporarily store results

        try:
            # Estimate overhead for each stage
            for stage in self.stages:
                logger.debug(f"[Build-MemScale] Estimating overhead for stage '{stage.name}'...")
                # Ensure estimate_actor_memory_overhead is available
                overhead = estimate_actor_memory_overhead(stage.callable, actor_kwargs={"config": stage.config})
                stage_overheads[stage.name] = overhead
                total_overhead += overhead
                logger.debug(f"[Build-MemScale] Stage '{stage.name}' overhead: {overhead / (1024 * 1024):.2f} MB")

            # Calculate average and required replicas based on threshold
            avg_overhead = total_overhead / len(self.stages) if self.stages else 0.0
            if avg_overhead <= 0:
                logger.warning(
                    "[Build-MemScale] Could not calculate positive average overhead; skipping max_replica adjustment."
                )
                self.stage_memory_overhead = stage_overheads  # Store estimates even if no adjustment
                return

            logger.debug(f"[Build-MemScale] Average overhead per stage: {avg_overhead / (1024 * 1024):.2f} MB")
            total_system_memory = psutil.virtual_memory().total
            threshold_bytes = self.dynamic_memory_threshold * total_system_memory
            logger.debug(
                f"[Build-MemScale] System Mem: {total_system_memory / (1024 * 1024):.1f}MB, "
                f"Threshold: {threshold_bytes / (1024 * 1024):.1f}MB ({self.dynamic_memory_threshold * 100:.0f}%)"
            )

            required_total_replicas = int(threshold_bytes / avg_overhead)
            current_total_replicas = sum(stage.max_replicas for stage in self.stages)
            logger.debug(
                f"[Build-MemScale] Current total max replicas: {current_total_replicas}; "
                f"Estimated required replicas for memory threshold: {required_total_replicas}"
            )

            # Adjust max_replicas proportionally if needed
            if required_total_replicas > 0 and current_total_replicas != required_total_replicas:
                ratio = required_total_replicas / current_total_replicas
                logger.info(f"[Build-MemScale] Adjusting max_replicas by scaling factor: {ratio:.2f}")
                for stage in self.stages:
                    original = stage.max_replicas
                    # Ensure adjustment respects min_replicas (at least 1 unless min is 0)
                    adjusted_max = max(
                        stage.min_replicas,
                        1 if stage.min_replicas > 0 else 0,  # Ensure at least 1 if min > 0
                        int(stage.max_replicas * ratio),
                    )
                    if adjusted_max != original:
                        stage.max_replicas = adjusted_max  # Modify StageInfo directly
                        logger.info(
                            f"[Build-MemScale] Stage '{stage.name}': max_replicas adjusted {original} -> {adjusted_max}"
                        )
            else:
                logger.info("[Build-MemScale] No max_replica adjustment needed based on average overhead.")

            # Store the final estimates
            self.stage_memory_overhead = stage_overheads

        except Exception as e:
            logger.error(
                f"[Build-MemScale] Error during dynamic memory overhead estimation: {e}. "
                "Skipping max_replica adjustment.",
                exc_info=True,
            )
            # Store any estimates calculated so far, even if incomplete/error
            self.stage_memory_overhead = stage_overheads

    def _configure_autoscalers(self) -> None:
        """
        Updates the PIDController cost estimates and ResourceConstraintManager max_replicas
        based on the current pipeline configuration (including dynamic adjustments).
        """
        logger.debug("[Build-Configure] Configuring autoscalers...")
        total_max_replicas = 0
        default_cost_bytes = 100 * 1024 * 1024  # Default if estimate missing

        for stage in self.stages:
            # Use potentially adjusted max_replicas
            total_max_replicas += stage.max_replicas
            # Use estimated overhead if available, otherwise default
            overhead_bytes = self.stage_memory_overhead.get(stage.name, default_cost_bytes)
            cost_mb = int(overhead_bytes / (1024 * 1024))
            # Ensure cost is at least 1MB for PID controller estimates
            self.pid_controller.stage_cost_estimates[stage.name] = max(1, cost_mb)

        # Update constraint manager with the final total max replicas allowed
        self.constraint_manager.max_replicas = total_max_replicas

        logger.info(f"[Build-Configure] Autoscalers configured. Total Max Replicas: {total_max_replicas}")
        logger.debug(f"[Build-Configure] PID stage cost estimates (MB): {self.pid_controller.stage_cost_estimates}")

    def _instantiate_initial_actors(self) -> Dict[str, List[Any]]:
        """
        Instantiates the initial set of Ray actors for each stage based on min_replicas.
        Populates self.stage_actors and self.scaling_state. Stats are handled by
        the separate RayStatsCollector.

        Requires self._structure_lock if called concurrently with other structure modifications.

        Returns:
            Dict[str, List[Any]]: The dictionary of created stage actors.
        """
        logger.info("[Build-Actors] Instantiating initial stage actors (min_replicas)...")
        created_actors: Dict[str, List[Any]] = {}
        new_scaling_state: Dict[str, str] = {}

        # Assume self.stages is populated correctly before this call
        current_stages = self.stages[:]  # Work on a copy if needed

        for stage in current_stages:
            # Check if stage has required attributes (adjust as needed)
            if not all(
                hasattr(stage, attr) for attr in ["name", "min_replicas", "is_source", "is_sink", "callable", "config"]
            ):
                logger.error(
                    f"[Build-Actors] Stage '{getattr(stage, 'name', 'Unknown')}' is missing required attributes."
                    f" Skipping actor creation."
                )
                continue  # Or raise an error

            replicas = []
            # Determine initial count: min_replicas, but at least 1 for source/sink
            num_initial_actors = stage.min_replicas
            if stage.is_source or stage.is_sink:
                num_initial_actors = max(1, stage.min_replicas)

            if num_initial_actors == 0:
                logger.debug(f"[Build-Actors] Stage '{stage.name}' has min_replicas=0, creating 0 initial actors.")
            else:
                logger.debug(f"[Build-Actors] Stage '{stage.name}' creating {num_initial_actors} initial actor(s).")

            # Create actors
            for i in range(num_initial_actors):
                # Generate unique actor name (assuming ray allows this naming scheme)
                actor_name = f"{stage.name}_{uuid.uuid4()}"
                logger.debug(
                    f"[Build-Actors] Creating actor '{actor_name}' ({i + 1}/{num_initial_actors})"
                    f" for stage '{stage.name}'"
                )
                try:
                    # Make sure stage.callable is actually a Ray remote class/function
                    if not hasattr(stage.callable, "options") or not hasattr(stage.callable, "remote"):
                        raise TypeError(f"Stage '{stage.name}' callable is not a Ray remote function/class.")

                    # Pass necessary options like name, concurrency
                    actor = stage.callable.options(name=actor_name, max_concurrency=100).remote(
                        config=stage.config,
                    )
                    replicas.append(actor)
                except Exception as e:
                    logger.error(
                        f"[Build-Actors] Failed to create actor '{actor_name}' for stage '{stage.name}': {e}",
                        exc_info=True,
                    )
                    # Propagate error to halt the build
                    raise RuntimeError(
                        f"Failed to build pipeline: actor creation error for stage '{stage.name}'"
                    ) from e

            # Store actors and initialize scaling state for the stage
            created_actors[stage.name] = replicas
            new_scaling_state[stage.name] = "Idle"

            logger.debug(f"[Build-Actors] Stage '{stage.name}' initial actors created: count={len(replicas)}")

        # --- Update instance variables under lock ---
        with self._structure_lock:
            self.stage_actors = created_actors
            self.scaling_state = new_scaling_state

        logger.info("[Build-Actors] Initial actor instantiation complete.")
        return created_actors

    def _create_and_wire_edges(self) -> List[ray.ObjectRef]:
        """
        Creates distributed queues for pipeline edges and wires up actor inputs/outputs.
        Populates self.edge_queues. Requires structure lock.

        Returns:
            List[ray.ObjectRef]: A list of Ray futures for the wiring calls.
        """
        logger.info("[Build-Wiring] Creating and wiring edges between stages...")
        wiring_refs = []
        new_edge_queues: Dict[str, Tuple[Any, int]] = {}  # Build locally first

        # --- Acquire Lock for Reading structure and Writing edge_queues ---
        with self._structure_lock:
            # Read snapshots of structures needed
            current_connections = self.connections.copy()  # Assuming shallow copy is ok
            current_stage_actors = {name: list(actors) for name, actors in self.stage_actors.items()}

            for from_stage_name, connections_list in current_connections.items():
                for to_stage_name, queue_size in connections_list:
                    queue_name = f"{from_stage_name}_to_{to_stage_name}"
                    logger.debug(
                        f"[Build-Wiring] Creating queue '{queue_name}' (size {queue_size}) and preparing wiring."
                    )

                    try:
                        # Create the distributed queue
                        edge_queue = Queue(maxsize=queue_size)
                        new_edge_queues[queue_name] = (edge_queue, queue_size)

                        # Prepare wiring refs using the snapshots
                        source_actors = current_stage_actors.get(from_stage_name, [])
                        for actor in source_actors:
                            if hasattr(actor, "set_output_queue") and hasattr(actor.set_output_queue, "remote"):
                                wiring_refs.append(actor.set_output_queue.remote(edge_queue))
                            else:
                                logger.warning(
                                    f"[Build-Wiring] Actor in stage '{from_stage_name}'"
                                    f" missing remote set_output_queue method."
                                )

                        dest_actors = current_stage_actors.get(to_stage_name, [])
                        for actor in dest_actors:
                            if hasattr(actor, "set_input_queue") and hasattr(actor.set_input_queue, "remote"):
                                wiring_refs.append(actor.set_input_queue.remote(edge_queue))
                            else:
                                logger.warning(
                                    f"[Build-Wiring] Actor in stage '{to_stage_name}'"
                                    f" missing remote set_input_queue method."
                                )

                    except Exception as e:
                        logger.error(
                            f"[Build-Wiring] Failed to create or wire queue '{queue_name}': {e}", exc_info=True
                        )
                        raise RuntimeError(f"Failed to build pipeline: queue wiring error for '{queue_name}'") from e

            # --- Commit the new queues to the instance variable ---
            self.edge_queues = new_edge_queues
        # --- RELEASE LOCK ---

        logger.debug(f"[Build-Wiring] Submitted {len(wiring_refs)} wiring calls.")
        # Note: We return the refs, caller should wait on them if build needs to be synchronous.
        return wiring_refs

    @staticmethod
    def _wait_for_wiring(wiring_refs: List[ray.ObjectRef]) -> None:
        """
        Waits for the remote wiring calls (set_input/output_queue) to complete.
        """
        if not wiring_refs:
            logger.debug("[Build-WaitWiring] No wiring calls to wait for.")
            return

        logger.debug(f"[Build-WaitWiring] Waiting for {len(wiring_refs)} wiring calls to complete...")
        try:
            # Wait for all futures in the list
            ray.get(wiring_refs)
            logger.info("[Build-WaitWiring] All wiring calls completed successfully.")
        except Exception as e:
            # Handle potential errors during the remote calls
            logger.error(f"[Build-WaitWiring] Error during initial wiring confirmation: {e}", exc_info=True)
            raise RuntimeError("Failed to build pipeline: error confirming initial wiring") from e

    def add_source(
        self, *, name: str, source_actor: Any, config: BaseModel, min_replicas: int = 1, max_replicas: int = 1
    ) -> "RayPipeline":
        if min_replicas < 1:
            logger.warning(f"Source stage '{name}': min_replicas must be at least 1. Overriding to 1.")
            min_replicas = 1
        stage_info = StageInfo(
            name=name,
            callable=source_actor,
            config=config,
            is_source=True,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.stages.append(stage_info)
        logger.debug(f"Added source stage: {stage_info}")
        return self

    def add_stage(
        self, *, name: str, stage_actor: Any, config: BaseModel, min_replicas: int = 0, max_replicas: int = 1
    ) -> "RayPipeline":
        if min_replicas < 0:
            logger.warning(f"Stage '{name}': min_replicas cannot be negative. Overriding to 0.")
            min_replicas = 0
        stage_info = StageInfo(
            name=name, callable=stage_actor, config=config, min_replicas=min_replicas, max_replicas=max_replicas
        )
        self.stages.append(stage_info)
        logger.debug(f"Added intermediate stage: {stage_info}")
        return self

    def add_sink(
        self, *, name: str, sink_actor: Any, config: BaseModel, min_replicas: int = 1, max_replicas: int = 1
    ) -> "RayPipeline":
        if min_replicas < 0:
            logger.warning(f"Sink stage '{name}': min_replicas must be at least 0. Overriding to 0.")
            min_replicas = 0
        stage_info = StageInfo(
            name=name,
            callable=sink_actor,
            config=config,
            is_sink=True,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.stages.append(stage_info)
        logger.debug(f"Added sink stage: {stage_info}")
        return self

    # --- Method for defining connections ---
    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        if from_stage not in [s.name for s in self.stages]:
            logger.error(f"make_edge: Stage {from_stage} not found")
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            logger.error(f"make_edge: Stage {to_stage} not found")
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        logger.debug(f"Created edge from {from_stage} to {to_stage} with queue_size {queue_size}")
        return self

    # --- Build method ---
    def build(self) -> Dict[str, List[Any]]:
        """
        Builds the pipeline: performs dynamic scaling, configures controllers,
        instantiates initial actors, creates queues, and wires connections.

        Returns:
            Dict[str, List[Any]]: Dictionary mapping stage names to their initial list of actor handles.
                                   Returns an empty dict if build fails.
        """
        logger.info("--- Starting Pipeline Build Process ---")
        try:
            # Step 1: Perform optional dynamic memory scaling
            self._perform_dynamic_memory_scaling()

            # Step 2: Configure PID and Resource Manager based on final stage settings
            self._configure_autoscalers()

            # Step 3: Instantiate initial actors based on min_replicas
            # This populates self.stage_actors
            self._instantiate_initial_actors()

            # Step 4: Create queues and submit wiring calls
            wiring_futures = self._create_and_wire_edges()

            # Step 5: Wait for wiring calls to complete
            self._wait_for_wiring(wiring_futures)

            logger.info("--- Pipeline Build Completed Successfully ---")
            return self.stage_actors  # Return the populated actor dictionary

        except RuntimeError as e:
            # Catch errors raised explicitly by helper methods
            logger.critical(f"Pipeline build failed: {e}", exc_info=False)
            return {}
        except Exception as e:
            # Catch unexpected errors
            logger.critical(f"Unexpected error during pipeline build: {e}", exc_info=True)
            return {}

    # --- Scaling Logic ---
    def _create_single_replica(self, stage_info: StageInfo) -> Any:
        """Creates a single new Ray actor replica for the given stage."""
        actor_name = f"{stage_info.name}_{uuid.uuid4()}"
        logger.debug(f"[ScaleUtil] Creating new actor '{actor_name}' for stage '{stage_info.name}'")
        try:
            # Adjust .options() as needed (e.g., resource requests)
            new_actor = stage_info.callable.options(name=actor_name, max_concurrency=100).remote(
                config=stage_info.config
            )
            return new_actor
        except Exception as e:
            logger.error(
                f"[ScaleUtil] Failed to create actor '{actor_name}' for stage '{stage_info.name}':" f" {e}",
                exc_info=True,
            )
            # Propagate error to halt the scaling operation
            raise RuntimeError(f"Actor creation failed for stage '{stage_info.name}' during scale up") from e

    def _get_wiring_refs_for_actor(self, actor: Any, stage_name: str) -> List[ray.ObjectRef]:
        """Gets the Ray futures for wiring the input/output queues of a single actor."""
        wiring_refs = []
        # Wire outputs (if actor is a source for any connection)
        if stage_name in self.connections:
            for to_stage, _ in self.connections[stage_name]:
                queue_name = f"{stage_name}_to_{to_stage}"
                if queue_name in self.edge_queues and hasattr(actor, "set_output_queue"):
                    edge_queue, _ = self.edge_queues[queue_name]
                    logger.debug(f"[ScaleUtil] Wiring actor '{actor}' output to queue '{queue_name}'")
                    wiring_refs.append(actor.set_output_queue.remote(edge_queue))
                elif queue_name not in self.edge_queues:
                    logger.error(f"[ScaleUtil] Output queue '{queue_name}' not found for wiring actor '{actor}'")
                # No warning if actor missing method, handled during build wiring check

        # Wire inputs (if actor is a destination for any connection)
        for from_stage, conns in self.connections.items():
            for to_stage, _ in conns:
                if to_stage == stage_name:
                    queue_name = f"{from_stage}_to_{stage_name}"
                    if queue_name in self.edge_queues and hasattr(actor, "set_input_queue"):
                        edge_queue, _ = self.edge_queues[queue_name]
                        logger.debug(f"[ScaleUtil] Wiring actor '{actor}' input from queue '{queue_name}'")
                        wiring_refs.append(actor.set_input_queue.remote(edge_queue))
                    elif queue_name not in self.edge_queues:
                        logger.error(f"[ScaleUtil] Input queue '{queue_name}' not found for wiring actor '{actor}'")
                    # No warning if actor missing method

        return wiring_refs

    def _start_actors(self, actors_to_start: List[Any], stage_name: str) -> None:
        """Starts a list of actors if they have a 'start' method and waits for completion."""
        start_refs = []
        for actor in actors_to_start:
            if hasattr(actor, "start"):
                logger.debug(f"[ScaleUtil] Starting actor '{actor}' for stage '{stage_name}'")
                start_refs.append(actor.start.remote())

        if not start_refs:
            logger.debug(f"[ScaleUtil] No actors with start() method found for stage '{stage_name}'.")
            return

        logger.debug(f"[ScaleUtil] Waiting for {len(start_refs)} actor starts for stage '{stage_name}'...")
        try:
            ray.get(start_refs)
            logger.debug(f"[ScaleUtil] {len(start_refs)} actors started successfully for stage '{stage_name}'.")
        except Exception as e:
            logger.error(
                f"[ScaleUtil] Error waiting for actors to start for stage '{stage_name}':" f" {e}", exc_info=True
            )
            # Note: Actors might be started but confirmation failed. State might be inconsistent.
            # Consider raising an error to signal potential inconsistency?
            raise RuntimeError(f"Error confirming actor starts for stage '{stage_name}'") from e

    def _handle_scale_up(self, stage_info: StageInfo, current_count: int, target_count: int) -> None:
        """Handles the process of scaling a stage up."""
        stage_name = stage_info.name
        num_to_add = target_count - current_count
        logger.info(
            f"[ScaleUp-{stage_name}] Starting scale up from {current_count} to {target_count} replicas (+{num_to_add})."
        )
        self.scaling_state[stage_name] = "Scaling Up"

        new_actors = []
        all_wiring_refs = []

        try:
            # 1. Create all new actors first
            for _ in range(num_to_add):
                new_actor = self._create_single_replica(stage_info)
                new_actors.append(new_actor)

            # 2. Get wiring futures for all new actors
            for actor in new_actors:
                all_wiring_refs.extend(self._get_wiring_refs_for_actor(actor, stage_name))

            # 3. Wait for all wiring to complete (reuse build helper)
            self._wait_for_wiring(all_wiring_refs)  # Handles waiting and errors

            # 4. Start all newly wired actors
            self._start_actors(new_actors, stage_name)  # Handles waiting and errors

            # 5. Add successfully created, wired, and started actors to the official list
            self.stage_actors[stage_name].extend(new_actors)

            logger.info(
                f"[ScaleUp-{stage_name}] Scale up complete. New replica count: {len(self.stage_actors[stage_name])}"
            )

        except Exception as e:
            # Catch errors from create, wiring wait, or start wait
            logger.error(f"[ScaleUp-{stage_name}] Error during scale up process: {e}", exc_info=False)  # Log concisely
            self.scaling_state[stage_name] = "Error"
            # --- CRITICAL ---
            # Cleanup is hard here. Actors might be created but not wired/started.
            # Or wired but failed to start. Attempting ray.kill on 'new_actors' might
            # kill actors that shouldn't be killed if the error happened late.
            # Safest immediate action is often to log the error state and potentially
            # let the next scaling cycle try to reconcile. Manual intervention might be needed.
            logger.critical(
                f"[ScaleUp-{stage_name}] Failed to complete scale up. Manual cleanup of actors {new_actors}"
                f" might be required."
            )
            # Do NOT add potentially broken actors to self.stage_actors if error occurred before that point.
            # If error happened after adding them, the state is already inconsistent.

        finally:
            # Reset state only if the process didn't end in an error state
            if self.scaling_state.get(stage_name) == "Scaling Up":
                self.scaling_state[stage_name] = "Idle"

    @staticmethod
    def _stop_actors(actors_to_remove: List[Any], stage_name: str) -> None:
        """Initiates stop/kill for a list of actors."""
        stop_refs = []

        logger.debug(f"[ScaleDown-{stage_name}] Processing {len(actors_to_remove)} actors for removal.")
        for actor in actors_to_remove:
            if hasattr(actor, "stop"):
                logger.debug(f"[ScaleDown-{stage_name}] Calling stop() on actor '{actor}'.")
                try:
                    # Initiate stop, don't wait here
                    stop_refs.append(actor.stop.remote())
                except Exception as e:
                    # Log if submitting stop fails, might need kill
                    logger.error(
                        f"[ScaleDown-{stage_name}] Error submitting stop() for actor '{actor}':" f" {e}.",
                        exc_info=False,
                    )
            else:
                logger.warning(f"[ScaleDown-{stage_name}] Actor '{actor}' has no stop() method.")

        # Log how many stop requests were sent (they proceed in the background)
        if stop_refs:
            logger.debug(f"[ScaleDown-{stage_name}] {len(stop_refs)} stop() requests sent (async).")

    def _handle_scale_down(self, stage_name: str, current_replicas: List[Any], target_count: int) -> None:
        """Handles the process of scaling a stage down."""
        current_count = len(current_replicas)
        num_to_remove = current_count - target_count
        logger.info(
            f"[ScaleDown-{stage_name}] Starting scale down from {current_count} to {target_count}"
            f" replicas (-{num_to_remove})."
        )
        self.scaling_state[stage_name] = "Scaling Down"

        if num_to_remove <= 0:  # Should not happen if called correctly, but safety check
            logger.warning(f"[ScaleDown-{stage_name}] Invalid num_to_remove ({num_to_remove}). Aborting scale down.")
            self.scaling_state[stage_name] = "Idle"
            return

        # Identify actors to remove (e.g., from the end of the list)
        actors_to_remove = current_replicas[-num_to_remove:]
        remaining_actors = current_replicas[:-num_to_remove]

        # Update the official actor list *immediately*
        self.stage_actors[stage_name] = remaining_actors
        logger.debug(f"[ScaleDown-{stage_name}] Actor list updated. Remaining actors: {len(remaining_actors)}")

        # Initiate stop/kill for the removed actors
        self._stop_actors(actors_to_remove, stage_name)

        logger.info(f"[ScaleDown-{stage_name}] Scale down initiated. New target replica count: {len(remaining_actors)}")
        # Set state to Idle optimistically, assuming stop/kill initiated successfully
        self.scaling_state[stage_name] = "Idle"

    def _scale_stage(self, stage_name: str, new_replica_count: int) -> None:
        """
        Orchestrates scaling a stage up or down to the target replica count.

        Validates input, calculates the final target respecting bounds, and
        delegates the actual scaling work to helper methods.
        """
        logger.debug(f"[ScaleStage-{stage_name}] Request received for target count: {new_replica_count}")

        # --- Pre-checks and Setup ---
        current_replicas = self.stage_actors.get(stage_name, [])
        current_count = len(current_replicas)
        stage_info = next((s for s in self.stages if s.name == stage_name), None)

        if stage_info is None:
            logger.error(f"[ScaleStage-{stage_name}] Stage info not found. Cannot scale.")
            return  # Exit if stage info is missing

        # Calculate final target count, clamped by stage's min/max replicas
        target_count = max(stage_info.min_replicas, min(new_replica_count, stage_info.max_replicas))
        if target_count != new_replica_count:
            logger.debug(
                f"[ScaleStage-{stage_name}] Requested count {new_replica_count} adjusted to {target_count}"
                f" based on min/max bounds ({stage_info.min_replicas}/{stage_info.max_replicas})."
            )

        # --- Check if Scaling is Needed ---
        if target_count == current_count:
            logger.debug(
                f"[ScaleStage-{stage_name}] Already at target replica count ({current_count}). No action needed."
            )
            # Ensure state is Idle if no action is taken but it was previously scaling
            if self.scaling_state.get(stage_name) != "Idle":
                self.scaling_state[stage_name] = "Idle"
            return  # Exit if no change needed

        # --- Delegate to Handlers ---
        try:
            if target_count > current_count:
                self._handle_scale_up(stage_info, current_count, target_count)
            elif target_count < current_count:
                # Pass the original list before modification for clarity
                self._handle_scale_down(stage_name, list(current_replicas), target_count)
        except RuntimeError as e:
            # Catch errors specifically raised by handlers (e.g., actor creation failed)
            logger.error(f"[ScaleStage-{stage_name}] Scaling failed: {e}", exc_info=False)
            # State might be inconsistent, set to Error
            self.scaling_state[stage_name] = "Error"
        except Exception as e:
            # Catch unexpected errors during delegation
            logger.error(
                f"[ScaleStage-{stage_name}] Unexpected error during scaling orchestration:" f" {e}", exc_info=True
            )
            self.scaling_state[stage_name] = "Error"

    def _get_global_in_flight(self, stage_stats: Dict[str, Dict[str, int]]) -> int:
        """Calculates total in-flight items across all stages from a stats dict."""
        if not stage_stats:
            return 0
        return sum(data.get("in_flight", 0) for data in stage_stats.values())

    def _is_pipeline_quiet(self) -> bool:
        """
        Checks if the pipeline is considered "quiet" for potential queue flushing.
        Uses the RayStatsCollector for recent stats.
        """

        return False  # disable quiet check for testing

        if self._is_flushing:
            logger.debug("Pipeline quiet check: False (Flush already in progress)")
            return False

        time_since_last_flush = time.time() - self._last_queue_flush_time
        if time_since_last_flush < self.queue_flush_interval_seconds:
            # logger.debug(f"Pipeline quiet check: False (Too soon since last flush)")  # Frequent log
            return False

        # --- Get latest stats from the collector ---
        current_stage_stats, last_update_time, stats_were_successful = self.stats_collector.get_latest_stats()
        last_update_age = time.time() - last_update_time

        # --- Check 1: Were the last stats collected successfully? ---
        if not stats_were_successful:
            logger.warning(f"Pipeline quiet check: False (Stats collection failed {last_update_age:.1f}s ago).")
            return False

        # --- Check 2: Are the stats recent enough? ---
        # Use the interval configured in the pipeline __init__
        max_stats_age_for_quiet = max(10.0, self._stats_collection_interval_seconds * 2.5)
        if last_update_age > max_stats_age_for_quiet:
            logger.warning(
                f"Pipeline quiet check: False (Stats too old: {last_update_age:.1f}s > {max_stats_age_for_quiet:.1f}s)."
            )
            return False

        # --- Check 3: Is the activity level low enough? ---
        if not current_stage_stats:
            # This might happen briefly after start/stop or if collection consistently fails
            logger.warning("Pipeline quiet check: False (No stats currently available from collector).")
            return False

        global_in_flight = self._get_global_in_flight(current_stage_stats)
        is_quiet = global_in_flight <= self.quiet_period_threshold

        if is_quiet:
            logger.info(
                f"Pipeline IS quiet (Stats age: {last_update_age:.1f}s). In-Flight: {global_in_flight} <= "
                f"Threshold: {self.quiet_period_threshold}"
            )
        else:
            logger.debug(
                f"Pipeline quiet check: False (Activity: InFlight={global_in_flight} > "
                f"Threshold={self.quiet_period_threshold})"
            )

        return is_quiet

    def _wait_for_pipeline_drain(self, timeout_seconds: int) -> bool:
        """
        Actively monitors pipeline drain using direct calls to the stats collector.
        """
        start_time = time.time()
        logger.info(f"Waiting for pipeline drain (Timeout: {timeout_seconds}s)...")
        last_in_flight = -1
        drain_check_interval = 1.0  # Check every second

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= timeout_seconds:
                logger.warning(f"Pipeline drain timed out after {elapsed_time:.1f}s. Last In-Flight: {last_in_flight}")
                return False

            # --- Trigger immediate stats collection via the collector instance ---
            drain_stats = {}
            drain_success = False
            collection_error = None
            try:
                # Use the collector's method for a one-off, blocking collection
                drain_stats, drain_success = self.stats_collector.collect_stats_now()
            except Exception as e:
                logger.error(f"[DrainWait] Critical error during direct stats collection call: {e}.", exc_info=True)
                collection_error = e  # Indicate failure to even run collection

            # --- Process collection results ---
            global_in_flight = -1  # Default to unknown
            if not collection_error:
                # Use helper, works even if drain_success is False (partial stats)
                global_in_flight = self._get_global_in_flight(drain_stats)

            if global_in_flight != last_in_flight:
                status_msg = (
                    f"Collection Success: {drain_success}"
                    if not collection_error
                    else f"Collection Error: {type(collection_error).__name__}"
                )
                logger.info(
                    f"[DrainWait] Check at {elapsed_time:.1f}s: Global In-Flight={global_in_flight} ({status_msg})"
                )
                last_in_flight = global_in_flight

            # --- Check for successful drain ---
            # Requires BOTH in-flight=0 AND the collection reporting it was successful
            if global_in_flight == 0 and drain_success and not collection_error:
                logger.info(f"Pipeline confirmed drained (In-Flight=0) in {elapsed_time:.1f}s.")
                return True
            elif global_in_flight == 0:  # Saw zero, but collection wasn't fully successful
                logger.warning(
                    "[DrainWait] In-Flight reached 0, but stats collection had errors/timeouts."
                    " Cannot confirm drain yet."
                )

            # --- Wait ---
            remaining_time = timeout_seconds - elapsed_time
            sleep_duration = min(drain_check_interval, remaining_time, 1.0)  # Ensure positive sleep
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def _execute_queue_flush(self) -> bool:
        """
        Executes the queue flush procedure: pauses sources, waits for drain,
        creates new queues, re-wires actors, and resumes sources.
        Uses structure lock for safe access/modification of pipeline components.

        Returns
        -------
        bool
            True if flush completed successfully, False otherwise.
        """
        # Acquire lock early to prevent concurrent flushes? Or rely on _is_flushing flag?
        # Using the flag is simpler for now.
        if self._is_flushing:
            logger.warning("Queue flush requested but already in progress. Ignoring.")
            return False

        self._is_flushing = True  # Set flag immediately
        logger.info("--- Starting Queue Flush ---")
        overall_success = False
        source_actors_paused = []
        pause_refs = []
        resume_refs = []
        new_edge_queues: Optional[Dict[str, Tuple[Any, int]]] = None

        # --- Read initial pipeline structure under lock ---
        # Get consistent snapshots of what needs pausing/re-wiring
        current_stages: List[StageInfo] = []
        current_stage_actors: Dict[str, List[Any]] = {}
        current_edge_queues: Dict[str, Tuple[Any, int]] = {}
        current_connections: Dict[str, List[Tuple[str, int]]] = {}
        try:
            with self._structure_lock:
                current_stages = self.stages[:]
                current_stage_actors = {name: list(actors) for name, actors in self.stage_actors.items()}
                current_edge_queues = self.edge_queues.copy()
                current_connections = (
                    self.connections.copy()
                )  # Assuming connections dict structure is safe to copy shallowly

            # --- 1. Pause Source Stages ---
            logger.info("Pausing source stages...")
            pause_timeout = 60.0
            for stage in current_stages:  # Use the snapshot
                if stage.is_source:
                    # Use the snapshot of actors for this stage
                    actors = current_stage_actors.get(stage.name, [])
                    for actor in actors:
                        actor_repr = repr(actor)
                        if (
                            hasattr(actor, "pause")
                            and callable(getattr(actor, "pause"))
                            and hasattr(actor.pause, "remote")
                        ):
                            try:
                                pause_refs.append(actor.pause.remote())
                                source_actors_paused.append(actor)
                                logger.debug(f"Pause signal sent to source actor {actor_repr} for stage {stage.name}")
                            except Exception as e:
                                logger.error(f"Failed sending pause signal to {actor_repr}: {e}")
                        else:
                            logger.warning(f"Source actor {actor_repr} lacks remote 'pause' method. Skipping.")

            if pause_refs:
                logger.info(f"Waiting up to {pause_timeout}s for {len(pause_refs)} source actors to pause...")
                try:
                    # Use ray.get with timeout for simplicity if individual failures are acceptable to log
                    ray.get(pause_refs, timeout=pause_timeout)
                    logger.info(f"{len(pause_refs)} source actors acknowledged pause (or call completed).")
                except GetTimeoutError:
                    logger.warning(f"Timeout waiting for {len(pause_refs)} source actors to pause.")
                    # Proceed cautiously
                except Exception as e:
                    logger.error(f"Error waiting for source actors to pause: {e}. Proceeding cautiously.")

            # --- 2. Wait for Pipeline Drain ---
            # This method uses self.stats_collector.collect_stats_now() internally
            logger.info("Waiting for pipeline to drain...")
            drain_successful = self._wait_for_pipeline_drain(self.queue_flush_drain_timeout_seconds)

            if not drain_successful:
                logger.error("Pipeline drain failed or timed out. ABORTING queue flush.")
                raise RuntimeError("Pipeline drain failed, aborting flush.")  # Triggers finally block

            logger.info("Pipeline drain successful.")

            # --- 3. Create New Queues ---
            logger.info("Creating new replacement queues...")
            new_edge_queues = {}
            # Use the snapshot of edge queues from the beginning
            for queue_name, (_, queue_size) in current_edge_queues.items():
                try:
                    # Ensure Queue is the correct Ray type
                    new_queue = Queue(maxsize=queue_size)
                    new_edge_queues[queue_name] = (new_queue, queue_size)
                    logger.debug(f"Created new queue: {queue_name} (size: {queue_size})")
                except Exception as e:
                    logger.error(f"Failed to create new queue '{queue_name}': {e}. ABORTING queue flush.")
                    raise RuntimeError(f"Failed to create new queue '{queue_name}'.") from e

            # --- 4. Re-wire Actors to New Queues ---
            logger.info("Re-wiring actors to new queues...")
            wiring_refs = []
            wiring_timeout = 120.0
            # Use the snapshot of connections and actors
            for from_stage_name, conns in current_connections.items():
                for to_stage_name, _ in conns:
                    queue_name = f"{from_stage_name}_to_{to_stage_name}"
                    if queue_name not in new_edge_queues:
                        logger.error(f"Logic error: New queue missing for connection {queue_name}. ABORTING.")
                        raise RuntimeError(f"New queue missing for {queue_name}")

                    new_queue_actor, _ = new_edge_queues[queue_name]

                    # Re-wire source stage outputs (using actor snapshot)
                    for actor in current_stage_actors.get(from_stage_name, []):
                        actor_repr = repr(actor)
                        if (
                            hasattr(actor, "set_output_queue")
                            and callable(getattr(actor, "set_output_queue"))
                            and hasattr(actor.set_output_queue, "remote")
                        ):
                            try:
                                wiring_refs.append(actor.set_output_queue.remote(new_queue_actor))
                                logger.debug(f"Sent set_output_queue({queue_name}) to {actor_repr}")
                            except Exception as e:
                                logger.error(f"Failed sending set_output_queue to {actor_repr}: {e}")
                        else:
                            logger.warning(f"Actor {actor_repr} lacks remote set_output_queue method.")

                    # Re-wire destination stage inputs (using actor snapshot)
                    for actor in current_stage_actors.get(to_stage_name, []):
                        actor_repr = repr(actor)
                        if (
                            hasattr(actor, "set_input_queue")
                            and callable(getattr(actor, "set_input_queue"))
                            and hasattr(actor.set_input_queue, "remote")
                        ):
                            try:
                                wiring_refs.append(actor.set_input_queue.remote(new_queue_actor))
                                logger.debug(f"Sent set_input_queue({queue_name}) to {actor_repr}")
                            except Exception as e:
                                logger.error(f"Failed sending set_input_queue to {actor_repr}: {e}")
                        else:
                            logger.warning(f"Actor {actor_repr} lacks remote set_input_queue method.")

            if wiring_refs:
                logger.info(f"Waiting up to {wiring_timeout}s for {len(wiring_refs)} actors to re-wire...")
                try:
                    ready, not_ready = ray.wait(wiring_refs, num_returns=len(wiring_refs), timeout=wiring_timeout)
                    if not_ready:
                        logger.error(f"{len(not_ready)} actors failed re-wiring within {wiring_timeout}s. ABORTING.")
                        raise RuntimeError("Actor re-wiring timed out or failed.")
                    ray.get(ready)  # Check for exceptions within set_input/output
                    logger.info(f"{len(ready)} actors re-wired successfully.")
                except Exception as e:
                    logger.error(f"Error waiting for actors to re-wire: {e}. ABORTING.")
                    raise RuntimeError("Actor re-wiring failed.") from e

            # --- 5. Update Internal State (Commit Point) ---
            logger.info("Committing new queues to pipeline state.")
            # --- ACQUIRE LOCK for WRITE ---
            with self._structure_lock:
                logger.debug(f"Replacing {len(self.edge_queues)} old queues with {len(new_edge_queues)} new queues.")
                self.edge_queues = new_edge_queues  # Commit the change
                # Reset queue stats history if applicable (check if self.queue_stats exists)
                if hasattr(self, "queue_stats") and isinstance(self.queue_stats, dict):
                    for queue_name in list(self.queue_stats.keys()):
                        if queue_name not in self.edge_queues:
                            logger.debug(f"Removing old queue stats history for {queue_name}")
                            del self.queue_stats[queue_name]
                        else:
                            logger.debug(f"Resetting queue stats history for {queue_name}")
                            self.queue_stats[queue_name] = []
                else:
                    logger.debug("Skipping queue_stats reset (attribute not found or not a dict).")
            # --- RELEASE LOCK ---
            overall_success = True  # Mark success *after* commit

        except Exception as e:
            # Catch errors from Drain, Queue creation, Wiring
            logger.error(f"Error during queue flush procedure: {e}", exc_info=True)
            overall_success = False  # Ensure success is false if any step failed

        finally:
            # --- 6. Resume Source Stages (Always attempt) ---
            if source_actors_paused:
                logger.info(f"Attempting to resume {len(source_actors_paused)} source actors...")
                resume_timeout = 30.0
                resume_refs = []
                for actor in source_actors_paused:  # Use list of actors we actually paused
                    actor_repr = repr(actor)
                    if (
                        hasattr(actor, "resume")
                        and callable(getattr(actor, "resume"))
                        and hasattr(actor.resume, "remote")
                    ):
                        try:
                            resume_refs.append(actor.resume.remote())
                            logger.debug(f"Sent resume signal to {actor_repr}")
                        except Exception as e:
                            logger.error(f"Failed sending resume signal to {actor_repr}: {e}")
                    else:
                        logger.warning(f"Paused source actor {actor_repr} lacks remote 'resume' method.")

                if resume_refs:
                    logger.info(f"Waiting up to {resume_timeout}s for {len(resume_refs)} actors to resume...")
                    try:
                        # Don't need ready/not_ready separation as much here, just log errors
                        ray.get(resume_refs, timeout=resume_timeout)
                        logger.info(f"{len(resume_refs)} source actors resumed successfully (or call completed).")
                    except GetTimeoutError:
                        logger.warning(f"Timeout waiting for {len(resume_refs)} source actors to resume.")
                    except Exception as e:
                        logger.error(f"Error occurred while waiting for source actors to resume: {e}")

            # Update flush timestamp only if the core operation succeeded
            if overall_success:
                self._last_queue_flush_time = time.time()
                logger.info("--- Queue Flush Completed Successfully ---")
            else:
                logger.error("--- Queue Flush Failed ---")

            self._is_flushing = False  # Release the logical flush lock

        return overall_success

    def request_queue_flush(self, force: bool = False) -> None:
        logger.info(f"Manual queue flush requested (force={force}).")
        if self._is_flushing:
            logger.warning("Flush already in progress.")
            return
        if force or self._is_pipeline_quiet():
            self._execute_queue_flush()
        else:
            logger.info("Manual flush denied: pipeline not quiet or interval not met.")

    def _gather_controller_metrics(self, current_stage_stats: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Gathers current metrics for autoscaling controllers using provided stats.
        Requires locking for accessing pipeline structure (actors, stage info).
        """
        logger.debug("[ScalingMetrics] Gathering metrics for controllers...")
        current_stage_metrics: Dict[str, Dict[str, Any]] = {}
        global_in_flight = self._get_global_in_flight(current_stage_stats)
        logger.debug(f"[ScalingMetrics] Using Global In-Flight: {global_in_flight}")

        # Lock structure while accessing stages and actors
        with self._structure_lock:
            current_stages = self.stages[:]  # Copy list
            current_actors = {name: list(actors) for name, actors in self.stage_actors.items()}  # Copy dict and lists

        for stage in current_stages:
            stage_name = stage.name
            replicas = len(current_actors.get(stage_name, []))  # Use locked snapshot
            stats = current_stage_stats.get(stage_name, {"processing": 0, "in_flight": 0})
            if stage_name not in current_stage_stats:
                logger.warning(f"[ScalingMetrics] Stage '{stage_name}' missing from stats. Using defaults.")

            processing_count = stats.get("processing", 0)
            stage_in_flight = stats.get("in_flight", 0)
            queue_depth = max(0, stage_in_flight - processing_count)

            current_stage_metrics[stage_name] = {
                "replicas": replicas,
                "queue_depth": queue_depth,
                "processing": processing_count,
                "in_flight": stage_in_flight,
                "min_replicas": stage.min_replicas,
                "max_replicas": stage.max_replicas,
                "pipeline_in_flight": global_in_flight,
            }

        logger.debug(f"[ScalingMetrics] Gathered metrics for {len(current_stage_metrics)} stages.")
        return current_stage_metrics

    def _get_current_global_memory(self) -> int:
        """
        Safely retrieves the current global system memory usage (used, not free) in MB.
        Uses the previous measurement as a fallback only if the current read fails.

        Returns:
            int: Current global memory usage (RSS/used) in MB. Returns previous value
                 or 0 if the read fails and no previous value exists.
        """
        try:
            # psutil.virtual_memory().used provides total RAM used by processes
            current_global_memory_bytes = psutil.virtual_memory().used
            current_global_memory_mb = int(current_global_memory_bytes / (1024 * 1024))
            logger.debug(f"[ScalingMemCheck] Current global memory usage (used): {current_global_memory_mb} MB")
            return current_global_memory_mb
        except Exception as e:
            logger.error(
                f"[ScalingMemCheck] Failed to get current system memory usage: {e}. "
                f"Attempting to use previous value ({self.prev_global_memory_usage} MB).",
                exc_info=False,
            )
            # Use previous value if available, otherwise default to 0 (less ideal, but avoids None)
            # Returning 0 might incorrectly signal low memory usage if it's the first read that fails.
            return self.prev_global_memory_usage if self.prev_global_memory_usage is not None else 0

    def _calculate_scaling_adjustments(
        self, current_stage_metrics: Dict[str, Dict[str, Any]], current_global_memory_mb: int
    ) -> Dict[str, int]:
        """
        Runs the PID controller and Resource Constraint Manager (RCM) to determine
        the final target replica count for each stage based on current metrics and memory.

        Parameters:
            current_stage_metrics: Metrics gathered by _gather_controller_metrics.
            current_global_memory_mb: Current system memory usage from _get_current_global_memory.

        Returns:
            Dict[str, int]: Dictionary mapping stage name to its final target replica count
                            after applying PID logic and RCM constraints. Returns current
                            replica counts as targets if controllers fail.
        """
        logger.debug("[ScalingCalc] Calculating adjustments via PID and RCM...")
        num_edges = len(self.edge_queues)  # Get current edge count for RCM cost estimation

        try:
            # --- 1. Get Initial Proposals from PID Controller ---
            # PID controller uses queue depth, processing count etc. per stage
            initial_proposals = self.pid_controller.calculate_initial_proposals(current_stage_metrics)
            # Log the raw proposals from the PID before constraints
            proposed_counts = {name: proposal.proposed_replicas for name, proposal in initial_proposals.items()}
            logger.debug(f"[ScalingCalc] PID Initial Proposals (Target Replicas): {proposed_counts}")

            # --- 2. Apply Constraints using Resource Constraint Manager ---
            # RCM takes PID proposals and applies global constraints like total max replicas
            # and memory limits. It needs current memory and *previous* memory (if available)
            # to estimate impact of scaling actions.
            final_adjustments = self.constraint_manager.apply_constraints(
                initial_proposals=initial_proposals,
                current_global_memory_usage=current_global_memory_mb,
                num_edges=num_edges,
            )
            logger.debug(f"[ScalingCalc] RCM Final Adjustments (Target Replicas): {final_adjustments}")
            return final_adjustments

        except Exception as e:
            logger.error(f"[ScalingCalc] Error during controller execution: {e}", exc_info=True)
            # Fallback Strategy: If controllers fail, propose no change to avoid erratic behavior.
            logger.warning("[ScalingCalc] Falling back to maintaining current replica counts due to controller error.")
            # Create a dictionary with current replica counts as the target
            fallback_adjustments = {}
            for stage_name, metrics in current_stage_metrics.items():
                # Ensure 'replicas' key exists from _gather_controller_metrics
                fallback_adjustments[stage_name] = metrics.get("replicas", 0)
            logger.debug(f"[ScalingCalc] Fallback Adjustments (Maintain Current): {fallback_adjustments}")
            return fallback_adjustments

    def _apply_scaling_actions(self, final_adjustments: Dict[str, int]) -> None:
        """
        Applies the calculated scaling adjustments (target replica counts) by
        calling the appropriate scaling logic (e.g., `_scale_stage`) for each
        stage where the target count differs from the current count. Executes
        these actions concurrently using a ThreadPoolExecutor.

        Parameters:
            final_adjustments: Dictionary from stage name to target replica count,
                               as determined by `_calculate_scaling_adjustments`.
        """
        stages_needing_action = []

        # --- Identify stages that require scaling up or down ---
        for stage_name, target_replica_count in final_adjustments.items():
            # Ensure stage exists in current state
            if stage_name not in self.stage_actors:
                logger.warning(f"[ScalingApply] Cannot apply scaling for unknown stage '{stage_name}'. Skipping.")
                continue

            current_count = len(self.stage_actors.get(stage_name, []))
            # Ensure target is within configured min/max bounds for the stage (should be handled by RCM, but
            # double check)
            stage_info = next((s for s in self.stages if s.name == stage_name), None)
            if not stage_info:
                logger.error(f"[ScalingApply] StageInfo not found for '{stage_name}'. Cannot validate bounds.")
                continue

            # Clamp target count within the stage's absolute min/max limits
            clamped_target = max(stage_info.min_replicas, min(stage_info.max_replicas, target_replica_count))
            if clamped_target != target_replica_count:
                logger.warning(
                    f"[ScalingApply-{stage_name}] Target count {target_replica_count} was outside stage bounds"
                    f" ({stage_info.min_replicas}-{stage_info.max_replicas}). Clamped to {clamped_target}."
                )
                target_replica_count = clamped_target

            if target_replica_count != current_count:
                stages_needing_action.append((stage_name, target_replica_count))
                logger.info(
                    f"[ScalingApply-{stage_name}] Action required: Current={current_count}, "
                    f"Target={target_replica_count}"
                    f" (Min={stage_info.min_replicas}, Max={stage_info.max_replicas})"
                )
            # Optional: Reset scaling state if no action needed but state was non-Idle
            # elif self.scaling_state.get(stage_name) != "Idle":
            #    logger.debug(f"[ScalingApply-{stage_name}] No action needed, ensuring state is Idle.")
            #    self.scaling_state[stage_name] = "Idle" # Assuming 'Idle' state exists

        if not stages_needing_action:
            logger.debug("[ScalingApply] No scaling actions required in this cycle.")
            return

        # --- Execute scaling actions concurrently ---
        # Limit concurrency to avoid overwhelming Ray scheduling or resources
        max_workers = min(len(stages_needing_action), 8)  # Example: Limit concurrent scaling operations
        logger.info(
            f"[ScalingApply] Submitting {len(stages_needing_action)} scaling actions using {max_workers} workers..."
        )

        action_results = {}  # Store future results/exceptions

        # Using ThreadPoolExecutor to manage concurrent calls to _scale_stage (which uses Ray remote calls)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ScalingAction"
        ) as executor:
            # Submit tasks: executor.submit(fn, *args)
            # Assuming _scale_stage exists and handles adding/removing actors for a stage
            future_to_stage = {
                executor.submit(self._scale_stage, stage_name, target_count): stage_name
                for stage_name, target_count in stages_needing_action
            }

            # Wait for submitted scaling operations initiated in *this cycle* to complete
            # Add a timeout to prevent indefinite blocking if a scaling action hangs
            wait_timeout = 180.0  # Timeout for scaling actions to complete (adjust as needed)
            logger.debug(f"[ScalingApply] Waiting up to {wait_timeout}s for scaling actions to complete...")

            # Use as_completed to process results as they finish (optional, wait is simpler)
            # done, not_done = concurrent.futures.wait(future_to_stage.keys(), timeout=wait_timeout)

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_stage, timeout=wait_timeout):
                stage_name = future_to_stage[future]
                try:
                    # Get the result (or raise exception if the task failed)
                    result = future.result()  # _scale_stage should return success/failure or relevant info
                    action_results[stage_name] = {"status": "completed", "result": result}
                    logger.debug(f"[ScalingApply-{stage_name}] Action completed. Result: {result}")
                except TimeoutError:  # Catch timeout from as_completed itself
                    logger.error(
                        f"[ScalingApply-{stage_name}] Scaling action timed out after {wait_timeout}s (as_completed)."
                    )
                    action_results[stage_name] = {"status": "timeout"}
                except Exception as exc:
                    # Catch exceptions raised *within* the _scale_stage function
                    logger.error(f"[ScalingApply-{stage_name}] Action failed with exception: {exc}", exc_info=True)
                    action_results[stage_name] = {"status": "error", "exception": exc}

        # Final summary log
        completed_count = sum(1 for res in action_results.values() if res["status"] == "completed")
        error_count = sum(1 for res in action_results.values() if res["status"] == "error")
        timeout_count = sum(1 for res in action_results.values() if res["status"] == "timeout")
        logger.info(
            f"[ScalingApply] Scaling actions summary: {completed_count} completed, {error_count} errors, "
            f"{timeout_count} timeouts."
        )
        logger.debug("[ScalingApply] Scaling action submission and processing completed for this cycle.")

    def _perform_scaling_and_maintenance(self) -> None:
        """
        Orchestrates scaling/maintenance, using RayStatsCollector for stats.
        """
        logger.debug("--- Performing Scaling & Maintenance Cycle ---")

        cycle_start_time = time.time()

        if self._is_flushing:
            logger.debug("Skipping scaling cycle: Queue flush in progress.")
            return

        # --- Check for quietness for flushing (uses stats collector via helper) ---
        try:
            if self._is_pipeline_quiet():
                logger.info("Pipeline detected as quiet, initiating queue flush.")
                flush_success = self._execute_queue_flush()
                logger.info(f"Automatic queue flush completed. Success: {flush_success}")
                return  # Skip scaling if flush occurred
            else:
                logger.debug("Pipeline not quiet; proceeding with scaling logic.")
        except Exception as e:
            logger.error(f"Error during quiet check or flush: {e}. Skipping cycle.", exc_info=True)
            return

        # --- Get Latest Stats from Collector ---
        current_stage_stats, last_update_time, stats_were_successful = self.stats_collector.get_latest_stats()
        last_update_age = time.time() - last_update_time

        # --- Validate Stats for Scaling ---
        max_stats_age_for_scaling = max(15.0, self._stats_collection_interval_seconds * 2)

        if not current_stage_stats:
            logger.error("[Scaling] Cannot scale: No statistics available from collector. Skipping cycle.")
            return
        if not stats_were_successful:
            logger.warning(
                f"[Scaling] Cannot scale: stats collection failed {last_update_age:.1f}s ago. Skipping cycle."
            )
            return
        if last_update_age > max_stats_age_for_scaling:
            logger.warning(
                f"[Scaling] Proceeding with STALE stats (Age: {last_update_age:.1f}s > "
                f"Threshold: {max_stats_age_for_scaling:.1f}s)."
            )
            # Continue, but log the warning

        # --- Gather Metrics (uses validated stats) ---
        current_stage_metrics = self._gather_controller_metrics(current_stage_stats)
        if not current_stage_metrics:
            logger.error("[Scaling] Failed to gather controller metrics. Skipping calculations.")
            return

        # --- Get Memory Usage ---
        current_global_memory_mb = self._get_current_global_memory()

        # --- Calculate Scaling Adjustments ---
        final_adjustments = self._calculate_scaling_adjustments(current_stage_metrics, current_global_memory_mb)

        # --- Update Memory Usage *After* Decision ---
        self.prev_global_memory_usage = current_global_memory_mb

        # --- Apply Scaling Actions ---
        self._apply_scaling_actions(final_adjustments)

        cycle_duration = time.time() - cycle_start_time
        logger.debug(f"--- Scaling & Maintenance Cycle Complete (Duration: {cycle_duration:.2f}s) ---")

    # --- Monitoring Thread ---

    def _get_monitor_data(self) -> List[Tuple]:
        """
        Fetches the latest statistics via RayStatsCollector and formats them
        for display. Uses structure lock for safe access to pipeline components.

        Returns
        -------
        List[Tuple]
            A list of tuples, where each tuple represents a row in the display table.
        """
        output_rows = []

        # --- Get latest stats from the collector ---
        current_stage_stats, last_update_time, stats_were_successful = self.stats_collector.get_latest_stats()
        last_update_age = time.time() - last_update_time

        # --- Get consistent snapshots of pipeline structure under lock ---
        with self._structure_lock:
            current_stages = self.stages[:]
            current_stage_actors = {name: list(actors) for name, actors in self.stage_actors.items()}
            current_edge_queues = self.edge_queues.copy()
            current_scaling_state = self.scaling_state.copy()
            current_is_flushing = self._is_flushing  # Read boolean flag under lock

        # --- Check stats staleness/failure ---
        # Use interval stored in pipeline for consistency in display rules
        max_stats_age_display = max(10.0, self._stats_collection_interval_seconds * 2.5)
        stats_stale = last_update_age > max_stats_age_display

        if not stats_were_successful or stats_stale:
            status = "Failed" if not stats_were_successful else "Stale"
            age_str = f"{last_update_age:.1f}s ago"
            warning_msg = f"[bold red]Stats {status} ({age_str})[/bold red]"
            output_rows.append((warning_msg, "", "", "", "", ""))  # Placeholder cols

        # --- Format data for each stage using snapshots ---
        for stage in current_stages:
            stage_name = stage.name
            # Get replica info from snapshot
            replicas = current_stage_actors.get(stage_name, [])
            replicas_str = f"{len(replicas)}/{stage.max_replicas}"  # Assuming stage has max_replicas
            if stage.min_replicas > 0:
                replicas_str += f" (min {stage.min_replicas})"

            # Use the stats collected earlier (no lock needed here)
            stats = current_stage_stats.get(stage_name, {"processing": 0, "in_flight": 0})
            processing_count = stats.get("processing", 0)
            stage_in_flight = stats.get("in_flight", 0)
            stage_queue_depth = max(0, stage_in_flight - processing_count)

            # Calculate Occupancy string (using edge queue snapshot)
            input_edges = [ename for ename in current_edge_queues if ename.endswith(f"_to_{stage_name}")]
            occupancy_str = "N/A"
            if input_edges:
                try:
                    first_q_name = input_edges[0]
                    # Get max size from the queue snapshot
                    _, max_q = current_edge_queues[first_q_name]
                    occupancy_str = f"{stage_queue_depth}/{max_q}"
                    if len(input_edges) > 1:
                        occupancy_str += " (multi)"
                except KeyError:
                    occupancy_str = f"{stage_queue_depth}/ERR"  # Queue missing from snapshot
                except Exception:
                    occupancy_str = f"{stage_queue_depth}/?"  # Other error
            elif stage.is_source:  # Assuming stage has is_source
                occupancy_str = "(Source)"

            # Get scaling state from snapshot
            scaling_state = current_scaling_state.get(stage_name, "Idle")

            output_rows.append(
                (stage_name, replicas_str, occupancy_str, scaling_state, str(processing_count), str(stage_in_flight))
            )

        # --- Add Total Pipeline Summary Row ---
        # Use stats collected earlier and flushing state from snapshot
        global_processing = sum(s.get("processing", 0) for s in current_stage_stats.values())
        global_in_flight = self._get_global_in_flight(current_stage_stats)  # Uses collected stats
        is_flushing_str = str(current_is_flushing)  # Use value read under lock

        output_rows.append(
            (
                "[bold]Total Pipeline[/bold]",
                "",
                "",  # Replica/Occupancy cols empty
                f"Flushing: {is_flushing_str}",  # State col
                f"[bold]{global_processing}[/bold]",  # Processing col
                f"[bold]{global_in_flight}[/bold]",  # In-Flight col
            )
        )

        return output_rows

    def _monitor_pipeline_loop(self, poll_interval: float) -> None:
        """
        Main loop for the monitoring thread. Fetches formatted data using
        `_get_monitor_data` (which reads shared stats) and updates the
        chosen display (Rich Console or GUI).

        Parameters
        ----------
        poll_interval : float
            The target interval in seconds between display updates.
        """
        thread_name = threading.current_thread().name
        logger.debug(
            f"{thread_name}: Monitoring thread started (Mode: {'GUI' if self.use_gui else 'Console'}, "
            f"Interval: {poll_interval}s)."
        )

        display_initialized = False

        # --- GUI Mode ---
        if self.use_gui:
            try:
                # Attempt to create GUI instance - requires display environment
                self._display_instance = GuiUtilizationDisplay(refresh_rate_ms=int(poll_interval * 1000))
                display_initialized = True

                # Start the blocking GUI loop, passing the data fetching function
                # This function (_get_monitor_data) will be called by the GUI's internal timer.
                logger.info(f"{thread_name}: Starting GUI display loop...")
                # The start method blocks until the GUI is closed.
                self._display_instance.start(self._get_monitor_data)
                logger.info(f"{thread_name}: GUI display loop finished.")

            except Exception as e:
                logger.error(f"{thread_name}: GUI initialization or execution failed: {e}", exc_info=True)
                logger.warning(f"{thread_name}: Falling back to console monitoring.")
                self.use_gui = False  # Disable GUI flag internally for this run
                self._display_instance = None  # Clear potentially failed instance
                display_initialized = False  # Ensure console mode runs below
                # No need for recursive call here, the console logic below will run

        # --- Console Mode (Rich) ---
        if not self.use_gui:
            try:
                # Create Rich display instance only if not already done (or if GUI failed)
                if not self._display_instance:
                    self._display_instance = UtilizationDisplay(refresh_rate=poll_interval)
                    # Start the Rich Live display context manager or similar non-blocking start
                    self._display_instance.start()
                    display_initialized = True
                    logger.info(f"{thread_name}: Started Rich console display.")

                # Console loop continues as long as monitoring flag is set
                while self._monitoring and display_initialized:
                    loop_start_time = time.time()
                    monitor_data = []  # Default empty
                    try:
                        # Fetch data using the method that reads shared stats
                        monitor_data = self._get_monitor_data()

                        # Update the Rich display if it exists and has update method
                        if self._display_instance and hasattr(self._display_instance, "update"):
                            self._display_instance.update(monitor_data)
                        else:
                            # Should not happen if display_initialized is True
                            logger.warning(f"{thread_name}: Display instance missing or lacks update method.")
                            break  # Exit loop if display is broken

                    except Exception as e:
                        logger.error(
                            f"{thread_name}: Error in console monitoring loop (data fetch or update): {e}",
                            exc_info=True,
                        )
                        # Continue loop, maybe the error is transient

                    # --- Calculate sleep time ---
                    elapsed = time.time() - loop_start_time
                    # Use the configured poll interval
                    sleep_time = max(0.1, poll_interval - elapsed)  # Minimum sleep

                    # Check flag *before* sleeping
                    if not self._monitoring:
                        break

                    # logger.debug(f"{thread_name}: Console monitor sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"{thread_name}: Error setting up or running console display loop: {e}", exc_info=True)
            finally:
                # Stop Rich display if it was initialized
                if display_initialized and self._display_instance and hasattr(self._display_instance, "stop"):
                    try:
                        logger.info(f"{thread_name}: Stopping Rich console display...")
                        self._display_instance.stop()
                    except Exception as e:
                        logger.error(f"{thread_name}: Error stopping Rich display: {e}")

        # --- Cleanup ---
        self._display_instance = None  # Clear display instance reference
        logger.debug(f"{thread_name}: Monitoring thread loop finished.")

    # --- Lifecycle Methods for Monitoring/Scaling Threads ---
    def _start_queue_monitoring(self, poll_interval: float = 5.0) -> None:
        if not self._monitoring:
            self._monitoring = True
            # Pass interval to the unified loop function
            self._monitor_thread = threading.Thread(
                target=self._monitor_pipeline_loop, args=(poll_interval,), daemon=True
            )
            self._monitor_thread.start()
            logger.info(f"Monitoring thread launched (Interval: {poll_interval}s).")

    def _stop_queue_monitoring(self) -> None:
        if self._monitoring:
            logger.debug("Stopping monitoring thread...")
            self._monitoring = False  # Signal loop to stop

            # If using GUI, explicitly call stop to destroy the window from this thread
            if self.use_gui and self._display_instance and hasattr(self._display_instance, "stop"):
                logger.debug("Requesting GUI stop...")
                try:
                    self._display_instance.stop()  # This signals the Tk mainloop to exit
                except Exception as e:
                    logger.error(f"Error stopping GUI display instance: {e}", exc_info=True)

            # Join the thread (will wait for Rich loop to finish or GUI mainloop to exit)
            if self._monitor_thread is not None:
                join_timeout = 10.0 if self.use_gui else 5.0  # Allow longer timeout for GUI shutdown
                self._monitor_thread.join(timeout=join_timeout)
                if self._monitor_thread.is_alive():
                    logger.warning("Monitoring thread did not exit cleanly.")
            self._monitor_thread = None
            self._display_instance = None  # Clear display instance ref
            logger.info("Monitoring stopped.")

    def _scaling_loop(self, interval: float) -> None:
        """Main loop for the scaling thread."""
        logger.info(f"Scaling loop started. Interval: {interval}s")
        while self._scaling_monitoring:
            try:
                self._perform_scaling_and_maintenance()
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}", exc_info=True)

            sleep_time = interval
            if not self._scaling_monitoring:
                break
            time.sleep(sleep_time)
        logger.info("Scaling loop finished.")

    def _start_scaling(self, poll_interval: float = 10.0) -> None:
        if not self._scaling_monitoring:
            self._scaling_monitoring = True
            self._scaling_thread = threading.Thread(target=self._scaling_loop, args=(poll_interval,), daemon=True)
            self._scaling_thread.start()
            logger.info(f"Scaling/Maintenance thread launched (Interval: {poll_interval}s).")

    def _stop_scaling(self) -> None:
        if self._scaling_monitoring:
            logger.debug("Stopping scaling/maintenance thread...")
            self._scaling_monitoring = False
            if self._scaling_thread is not None:
                self._scaling_thread.join(timeout=15)  # Allow more time for scaling actions
                if self._scaling_thread.is_alive():
                    logger.warning("Scaling thread did not exit cleanly.")
            self._scaling_thread = None
            logger.info("Scaling/Maintenance stopped.")

    # --- Pipeline Start/Stop ---
    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Start the pipeline: start actors, monitoring, and scaling.
        """
        # Check if built (optional, depends on design)
        with self._structure_lock:
            if not self.stage_actors:
                logger.error("Cannot start pipeline: Build() must be called first or pipeline is empty.")
                return

        logger.info("Starting pipeline execution...")
        start_refs = []
        # Lock structure while getting actors to start
        with self._structure_lock:
            for stage_name, actors in self.stage_actors.items():
                for actor in actors:
                    if hasattr(actor, "start"):
                        # Ensure start method is remote-callable
                        if hasattr(actor.start, "remote"):
                            start_refs.append(actor.start.remote())
                        else:
                            logger.warning(f"Actor in stage {stage_name} has 'start' but it's not remote. Skipping.")

        if start_refs:
            logger.debug(f"Waiting for {len(start_refs)} actors to start...")
            try:
                ray.get(start_refs, timeout=60.0)  # Add timeout
                logger.info(f"{len(start_refs)} actors started successfully.")
            except GetTimeoutError:
                logger.error(f"Timeout waiting for {len(start_refs)} actors to start.")
                self.stop()
                raise RuntimeError("Failed to start pipeline: actors did not start within timeout.")
            except Exception as e:
                logger.error(f"Error during actor start confirmation: {e}", exc_info=True)
                self.stop()  # Attempt cleanup
                raise RuntimeError("Failed to start pipeline: error confirming actor starts.") from e

        # Start background threads AFTER actors are confirmed running
        self.stats_collector.start()
        self._start_queue_monitoring(poll_interval=monitor_poll_interval)
        self._start_scaling(poll_interval=scaling_poll_interval)
        logger.info("Pipeline started successfully.")

    def stop(self) -> None:
        """
        Stop the pipeline: stop background threads and all stage actors.
        """
        logger.info("Stopping pipeline...")

        # 1. Stop background threads first to prevent further actions
        self._stop_scaling()
        self._stop_queue_monitoring()
        self.stats_collector.stop()

        # 2. Stop actors (graceful first, then kill)
        logger.debug("Stopping all stage actors...")
        stop_refs_map: Dict[ray.ObjectRef, Any] = {}  # Map ref back to actor for targeted kill
        actors_to_kill = []
        # Lock structure while getting actors to stop
        with self._structure_lock:
            current_actors = {name: list(actors) for name, actors in self.stage_actors.items()}

        for stage_name, actors in current_actors.items():
            for actor in actors:
                can_stop_gracefully = False
                if hasattr(actor, "stop") and hasattr(actor.stop, "remote"):
                    try:
                        ref = actor.stop.remote()
                        stop_refs_map[ref] = actor
                        can_stop_gracefully = True
                    except Exception as e:
                        logger.warning(
                            f"Error initiating stop for actor {actor} in stage {stage_name}: {e}. Will kill."
                        )
                        actors_to_kill.append(actor)
                if not can_stop_gracefully:
                    actors_to_kill.append(actor)  # Add actors without remote stop()

        if stop_refs_map:
            stop_refs = list(stop_refs_map.keys())
            logger.debug(f"Waiting for {len(stop_refs)} actors to stop gracefully...")
            try:
                ready, not_ready = ray.wait(stop_refs, num_returns=len(stop_refs), timeout=60.0)
                if not_ready:
                    logger.warning(
                        f"Timeout waiting for {len(not_ready)} actors to stop gracefully. Will proceed to kill."
                    )
                    # Add actors that timed out to the kill list
                    for ref in not_ready:
                        timed_out_actor = stop_refs_map.get(ref)
                        if timed_out_actor and timed_out_actor not in actors_to_kill:
                            actors_to_kill.append(timed_out_actor)
                logger.info(f"{len(ready)} actors stopped via stop().")
            except Exception as e:
                logger.error(f"Error during actor stop confirmation: {e}", exc_info=True)
                # Add all actors we tried to stop gracefully to the kill list on wait error
                actors_to_kill.extend(a for a in stop_refs_map.values() if a not in actors_to_kill)

        if actors_to_kill:
            logger.debug(f"Killing {len(actors_to_kill)} actors...")
            killed_count = 0
            for actor in actors_to_kill:
                try:
                    ray.kill(actor, no_restart=True)
                    killed_count += 1
                except Exception as e:
                    # Actor might already be dead
                    logger.warning(f"Failed or unnecessary attempt to kill actor {actor}: {e}")
            logger.debug(f"Kill attempted for {len(actors_to_kill)} actors (success count may vary).")

        # Clear internal state after stopping
        with self._structure_lock:
            self.stage_actors.clear()
            # Optionally clear edge_queues if they should be terminated/recreated

        logger.info("Pipeline stopped.")
