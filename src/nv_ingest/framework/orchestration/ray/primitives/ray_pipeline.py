# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import psutil
import uuid
import ray
from ray.util.queue import Queue
from typing import Dict, Optional, List, Tuple, Any
from pydantic import BaseModel
import concurrent.futures
import logging
import time

from nv_ingest.framework.orchestration.ray.util.pipeline.pid_controller import PIDController, ResourceConstraintManager
from nv_ingest.framework.orchestration.ray.util.system_tools.memory import estimate_actor_memory_overhead
from nv_ingest.framework.orchestration.ray.util.system_tools.visualizers import (
    GuiUtilizationDisplay,
    UtilizationDisplay,
)

logger = logging.getLogger(__name__)


# Placeholder for StageInfo if not defined
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
    """

    def __init__(
        self,
        # --- ADD GUI FLAG ---
        gui: bool = False,
        # --- General Pipeline Config ---
        dynamic_memory_scaling: bool = False,
        dynamic_memory_threshold: float = 0.75,
        queue_flush_interval_seconds: int = 600,
        queue_flush_drain_timeout_seconds: int = 300,
        quiet_period_threshold: int = 5,
        # --- PID Controller Config ---
        pid_kp: float = 0.1,
        pid_ki: float = 0.0075,
        pid_kd: float = 0.0,
        pid_target_queue_depth: int = 0,
        pid_penalty_factor: float = 0.1,
        pid_error_boost_factor: float = 1.5,
        pid_window_size: int = 10,
        pid_stage_memory_rate_threshold: float = 2000,
        # --- Resource Constraint Manager Config ---
        rcm_estimated_edge_cost_mb: int = 5000,
        rcm_memory_safety_buffer_fraction: float = 0.15,
    ) -> None:
        """
        Initializes the RayPipeline instance.

        Parameters
        ----------
        gui : bool, optional
            If True, attempts to launch a Tkinter GUI window for monitoring instead
            of using the console, by default False. Requires a display environment.
        # ... other parameters remain the same ...
        """
        self.use_gui = gui  # Store the flag

        # --- Core Pipeline Structure ---
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[Tuple[str, int]]] = {}
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Tuple[Queue, int]] = {}
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}

        # --- State ---
        self.scaling_state: Dict[str, str] = {}
        self.stage_stats: Dict[str, Dict[str, int]] = {}
        self.prev_global_memory_usage: Optional[int] = None

        # --- Build Time Config & State ---
        self.dynamic_memory_scaling = dynamic_memory_scaling
        self.dynamic_memory_threshold = dynamic_memory_threshold
        self.stage_memory_overhead: Dict[str, float] = {}

        # --- Background Threads ---
        self._monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._scaling_monitoring: bool = False
        self._scaling_thread: Optional[threading.Thread] = None
        self._display_instance: Optional[Any] = None  # Holds Rich or GUI display instance

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
        logger.info(
            f"PIDController initialized (kp={pid_kp}, ki={pid_ki}, kd={pid_kd}, targetQ={pid_target_queue_depth})"
        )
        try:
            total_system_memory_bytes = psutil.virtual_memory().total
            absolute_memory_threshold_mb = int(
                self.dynamic_memory_threshold * total_system_memory_bytes / (1024 * 1024)
            )
        except Exception as e:
            logger.error(f"Failed to get system memory: {e}. Using arbitrary high limit for memory threshold.")
            absolute_memory_threshold_mb = 1_000_000
        self.constraint_manager = ResourceConstraintManager(
            max_replicas=1,
            memory_threshold=absolute_memory_threshold_mb,
            estimated_edge_cost_mb=rcm_estimated_edge_cost_mb,
            memory_safety_buffer_fraction=rcm_memory_safety_buffer_fraction,
        )
        logger.info(
            f"ResourceConstraintManager initialized (MemThreshold={absolute_memory_threshold_mb}MB,"
            f" Buffer={rcm_memory_safety_buffer_fraction * 100:.1f}%)"
        )

        logger.info(f"GUI Mode Requested: {self.use_gui}")
        # Other logging remains the same

    def _perform_dynamic_memory_scaling(self) -> None:
        """
        Estimates stage memory overhead and adjusts max_replicas if dynamic scaling is enabled.
        Updates self.stage_memory_overhead and potentially modifies self.stages[...].max_replicas.
        """
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
                overhead = estimate_actor_memory_overhead(
                    stage.callable, actor_kwargs={"config": stage.config, "progress_engine_count": -1}
                )
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
        Populates self.stage_actors, self.scaling_state, self.stage_stats.

        Returns:
            Dict[str, List[Any]]: The dictionary of created stage actors.
        """
        logger.info("[Build-Actors] Instantiating initial stage actors (min_replicas)...")
        created_actors: Dict[str, List[Any]] = {}

        for stage in self.stages:
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
                # Generate unique actor name
                actor_name = f"{stage.name}_{uuid.uuid4()}"
                logger.debug(
                    f"[Build-Actors] Creating actor '{actor_name}' ({i + 1}/{num_initial_actors})"
                    f" for stage '{stage.name}'"
                )
                try:
                    # Assume actor takes 'config' and 'progress_engine_count'
                    # Adjust .options() as needed (e.g., resource requests)
                    actor = stage.callable.options(name=actor_name, max_concurrency=100).remote(
                        config=stage.config, progress_engine_count=-1
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

            # Store actors and initialize state for the stage
            created_actors[stage.name] = replicas
            self.scaling_state[stage.name] = "Idle"
            self.stage_stats[stage.name] = {"processing": 0, "in_flight": 0}
            logger.debug(f"[Build-Actors] Stage '{stage.name}' initial actors created: count={len(replicas)}")

        self.stage_actors = created_actors  # Update the instance variable
        return created_actors  # Return for consistency

    def _create_and_wire_edges(self) -> List[ray.ObjectRef]:
        """
        Creates distributed queues for pipeline edges and wires up actor inputs/outputs.
        Populates self.edge_queues and self.queue_stats.

        Returns:
            List[ray.ObjectRef]: A list of Ray futures for the wiring calls.
        """
        logger.info("[Build-Wiring] Creating and wiring edges between stages...")
        wiring_refs = []  # Collect futures for wiring calls

        for from_stage_name, connections_list in self.connections.items():
            for to_stage_name, queue_size in connections_list:
                queue_name = f"{from_stage_name}_to_{to_stage_name}"
                logger.debug(f"[Build-Wiring] Creating queue '{queue_name}' (size {queue_size}) and wiring actors.")

                try:
                    # Create the distributed queue
                    edge_queue = Queue(maxsize=queue_size)
                    self.edge_queues[queue_name] = (edge_queue, queue_size)
                    self.queue_stats[queue_name] = []  # Initialize stats tracking

                    # Wire output from source stage actors to this queue
                    source_actors = self.stage_actors.get(from_stage_name, [])
                    for actor in source_actors:
                        if hasattr(actor, "set_output_queue"):
                            # Call the remote method, store the future
                            wiring_refs.append(actor.set_output_queue.remote(edge_queue))
                        else:
                            logger.warning(
                                f"[Build-Wiring] Actor in stage '{from_stage_name}' missing set_output_queue method."
                            )

                    # Wire input to destination stage actors from this queue
                    dest_actors = self.stage_actors.get(to_stage_name, [])
                    for actor in dest_actors:
                        if hasattr(actor, "set_input_queue"):
                            # Call the remote method, store the future
                            wiring_refs.append(actor.set_input_queue.remote(edge_queue))
                        else:
                            logger.warning(
                                f"[Build-Wiring] Actor in stage '{to_stage_name}' missing set_input_queue method."
                            )

                except Exception as e:
                    logger.error(
                        f"[Build-Wiring] Failed to create or wire queue '{queue_name}':" f" {e}", exc_info=True
                    )
                    # Propagate error to halt the build
                    raise RuntimeError(f"Failed to build pipeline: queue wiring error for '{queue_name}'") from e

        logger.debug(f"[Build-Wiring] Submitted {len(wiring_refs)} wiring calls.")
        return wiring_refs

    def _wait_for_wiring(self, wiring_refs: List[ray.ObjectRef]) -> None:
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
            logger.critical(f"Pipeline build failed: {e}", exc_info=False)  # Log critical, don't need full
            # trace again
            # Optionally clean up partially created resources? Difficult with Ray actors.
            # Return empty dict or re-raise? Returning empty might be safer downstream.
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
            # Assume actor takes 'config' and 'progress_engine_count'
            # Adjust .options() as needed (e.g., resource requests)
            new_actor = stage_info.callable.options(name=actor_name, max_concurrency=100).remote(
                config=stage_info.config, progress_engine_count=-1
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

    def _stop_or_kill_actors(self, actors_to_remove: List[Any], stage_name: str) -> None:
        """Initiates stop/kill for a list of actors."""
        stop_refs = []
        actors_to_kill = []

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
                        f"[ScaleDown-{stage_name}] Error submitting stop() for actor '{actor}':"
                        f" {e}. Will attempt kill.",
                        exc_info=False,
                    )
                    actors_to_kill.append(actor)
            else:
                logger.warning(f"[ScaleDown-{stage_name}] Actor '{actor}' has no stop() method.")
                actors_to_kill.append(actor)

        # Kill actors that don't have stop() or failed during stop submission
        if actors_to_kill:
            logger.debug(f"[ScaleDown-{stage_name}] Killing {len(actors_to_kill)} actors.")
            for actor in actors_to_kill:
                logger.warning(f"[ScaleDown-{stage_name}] Killing actor '{actor}'.")
                try:
                    # no_restart=True ensures Ray doesn't try to revive it
                    ray.kill(actor, no_restart=True)
                except Exception as kill_e:
                    # Log error but continue trying to kill others
                    logger.error(f"[ScaleDown-{stage_name}] Failed to kill actor '{actor}': {kill_e}")

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
        self._stop_or_kill_actors(actors_to_remove, stage_name)

        logger.info(f"[ScaleDown-{stage_name}] Scale down initiated. New target replica count: {len(remaining_actors)}")
        # Set state to Idle optimistically, assuming stop/kill initiated successfully
        self.scaling_state[stage_name] = "Idle"

    # --- Refactored _scale_stage Method ---
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

    def _get_global_in_flight(self) -> int:
        global_in_flight = 0
        for stage_name in self.stage_stats:
            global_in_flight += self.stage_stats[stage_name].get("in_flight", 0)
        return global_in_flight

    def _update_stage_stats(self) -> None:
        """Fetches stats from actors and updates self.stage_stats. (FIXED QSIZE CALL)"""
        stage_stats_updates: Dict[str, Dict[str, int]] = {}
        stat_futures = {}
        # Launch get_stats calls in parallel
        for stage_name, actors in self.stage_actors.items():
            stage_stats_updates[stage_name] = {"processing": 0, "in_flight": 0}  # Initialize
            for actor in actors:
                if hasattr(actor, "get_stats"):
                    stat_futures[actor] = actor.get_stats.remote()

        # Collect actor stats results
        actor_stats_results = {}
        future_to_actor = {v: k for k, v in stat_futures.items()}
        ready, not_ready = ray.wait(list(stat_futures.values()), num_returns=len(stat_futures), timeout=2.0)

        for future in ready:
            actor = future_to_actor[future]
            try:
                actor_stats_results[actor] = ray.get(future)
            except Exception as e:
                logger.error(f"Error fetching stats result for actor {actor}: {e}")
                actor_stats_results[actor] = {"active_processing": 0}  # Default on error
        if not_ready:
            logger.warning(f"Timeout fetching stats for {len(not_ready)} actors.")
            for future in not_ready:
                actor = future_to_actor[future]
                actor_stats_results[actor] = {"active_processing": 0}  # Default on timeout

        # Aggregate stats per stage AND get queue sizes directly
        for stage in self.stages:
            stage_name = stage.name
            current_replicas = self.stage_actors.get(stage_name, [])
            processing_count = 0
            for actor in current_replicas:
                stats = actor_stats_results.get(actor, {"active_processing": 0})
                processing_count += int(stats.get("active_processing", 0))

            input_edges = [ename for ename in self.edge_queues if ename.endswith(f"_to_{stage_name}")]
            total_queued = 0
            for ename in input_edges:
                try:
                    # Access queue size safely and directly
                    queue_actor, _ = self.edge_queues[ename]
                    total_queued += queue_actor.qsize()
                except Exception as q_e:
                    # Log error but continue, maybe queue is terminating
                    logger.error(f"Error getting qsize for {ename}: {q_e}")

            # Recalculate in_flight based on actual processing and directly measured queue size
            stage_in_flight = processing_count + total_queued
            stage_stats_updates[stage_name]["processing"] = processing_count
            stage_stats_updates[stage_name]["in_flight"] = stage_in_flight

        # Update the main stats dictionary
        self.stage_stats = stage_stats_updates

    def _is_pipeline_quiet(self) -> bool:
        if self._is_flushing:
            return False
        time_since_last_flush = time.time() - self._last_queue_flush_time
        if time_since_last_flush < self.queue_flush_interval_seconds:
            return False

        self._update_stage_stats()  # Ensure fresh stats
        current_global_in_flight = self._get_global_in_flight()
        return current_global_in_flight <= self.quiet_period_threshold

    def _wait_for_pipeline_drain(self, timeout_seconds: int) -> bool:
        start_time = time.time()
        logger.info(f"Waiting for pipeline drain (timeout: {timeout_seconds}s)...")
        while time.time() - start_time < timeout_seconds:
            self._update_stage_stats()
            if self._get_global_in_flight() == 0:
                logger.info(f"Pipeline drained in {time.time() - start_time:.1f}s.")
                return True
            time.sleep(2)
        logger.warning(f"Pipeline drain timed out after {timeout_seconds}s.")
        return False

    def _execute_queue_flush(self) -> bool:
        if self._is_flushing:
            return False
        self._is_flushing = True
        logger.info("--- Starting Queue Flush ---")
        success = False
        source_actors_paused = []
        resume_refs = []  # Define here for potential use in finally/except

        try:
            # 1. Pause sources
            logger.info("Pausing source stages...")
            pause_refs = []
            for stage in self.stages:
                if stage.is_source:
                    actors = self.stage_actors.get(stage.name, [])
                    for actor in actors:
                        if hasattr(actor, "pause"):
                            pause_refs.append(actor.pause.remote())
                            source_actors_paused.append(actor)
            if pause_refs:
                ray.get(pause_refs)
                logger.info(f"{len(pause_refs)} source actors paused.")

            # 2. Wait for drain
            if not self._wait_for_pipeline_drain(self.queue_flush_drain_timeout_seconds):
                logger.error("Drain failed. Aborting flush.")
                # Rollback: Resume sources
                for actor in source_actors_paused:
                    if hasattr(actor, "resume"):
                        resume_refs.append(actor.resume.remote())
                if resume_refs:
                    ray.get(resume_refs)
                return False  # Use finally for self._is_flushing = False

            # 3. Create new queues
            logger.info("Creating new queues...")
            new_edge_queues: Dict[str, Tuple[Queue, int]] = {}
            for queue_name, (_, queue_size) in self.edge_queues.items():
                new_queue = Queue(maxsize=queue_size)
                new_edge_queues[queue_name] = (new_queue, queue_size)

            # 4. Re-wire actors
            logger.info("Re-wiring actors...")
            wiring_refs = []
            for from_stage_name, conns in self.connections.items():
                for to_stage_name, _ in conns:
                    queue_name = f"{from_stage_name}_to_{to_stage_name}"
                    if queue_name not in new_edge_queues:
                        continue  # Should not happen
                    new_queue, _ = new_edge_queues[queue_name]
                    # Re-wire outputs
                    for actor in self.stage_actors.get(from_stage_name, []):
                        if hasattr(actor, "set_output_queue"):
                            wiring_refs.append(actor.set_output_queue.remote(new_queue))
                    # Re-wire inputs
                    for actor in self.stage_actors.get(to_stage_name, []):
                        if hasattr(actor, "set_input_queue"):
                            wiring_refs.append(actor.set_input_queue.remote(new_queue))
            if wiring_refs:
                ray.get(wiring_refs)

            # 5. Update internal state
            self.edge_queues = new_edge_queues
            for queue_name in self.queue_stats:
                self.queue_stats[queue_name] = []

            # 6. Resume sources
            logger.info("Resuming source stages...")
            resume_refs = []  # Reset resume_refs
            for actor in source_actors_paused:
                if hasattr(actor, "resume"):
                    resume_refs.append(actor.resume.remote())
            if resume_refs:
                ray.get(resume_refs)

            self._last_queue_flush_time = time.time()
            logger.info("--- Queue Flush Completed Successfully ---")
            success = True

        except Exception as e:
            logger.error(f"Error during queue flush: {e}", exc_info=True)
            # Attempt emergency resume if sources were paused
            if source_actors_paused:
                logger.info("Attempting emergency resume of sources...")
                resume_refs = []
                for actor in source_actors_paused:
                    if hasattr(actor, "resume"):
                        resume_refs.append(actor.resume.remote())
                if resume_refs:
                    try:
                        ray.get(resume_refs)
                    except Exception as resume_e:
                        logger.error(f"Error during emergency resume: {resume_e}")
            success = False
        finally:
            self._is_flushing = False
        return success

    def request_queue_flush(self, force: bool = False) -> None:
        logger.info(f"Manual queue flush requested (force={force}).")
        if self._is_flushing:
            logger.warning("Flush already in progress.")
            return
        if force or self._is_pipeline_quiet():
            self._execute_queue_flush()
        else:
            logger.info("Manual flush denied: pipeline not quiet or interval not met.")

    def _gather_controller_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Gathers current metrics for each stage required by the autoscaling controllers.
        Relies on self.stage_stats being recently updated.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from stage name to its metrics dictionary.
        """
        logger.debug("[ScalingMetrics] Gathering metrics for controllers...")
        current_stage_metrics: Dict[str, Dict[str, Any]] = {}
        # Get global state once, used by all stages
        global_in_flight = self._get_global_in_flight()

        for stage in self.stages:
            stage_name = stage.name
            replicas = len(self.stage_actors.get(stage_name, []))
            # Use the freshly updated stats, default if missing
            stats = self.stage_stats.get(stage_name, {"processing": 0, "in_flight": 0})

            # Calculate queue depth based on stats
            processing_count = stats.get("processing", 0)
            stage_in_flight = stats.get("in_flight", 0)
            queue_depth = max(0, stage_in_flight - processing_count)  # Ensure non-negative

            # Assemble metrics dictionary for this stage
            current_stage_metrics[stage_name] = {
                "replicas": replicas,
                "queue_depth": queue_depth,
                "processing": processing_count,
                "in_flight": stage_in_flight,
                # Pass config limits needed by controllers/bounds checks
                "min_replicas": stage.min_replicas,
                "max_replicas": stage.max_replicas,
                # Pass global state if needed (e.g., for RCM bounds)
                "pipeline_in_flight": global_in_flight,
                # memory_usage could be added here if reliably measured per stage
            }
            logger.debug(
                f"[ScalingMetrics-{stage_name}] R={replicas}, Q={queue_depth}, Proc={processing_count},"
                f" InF={stage_in_flight}"
            )

        return current_stage_metrics

    def _get_current_global_memory(self) -> int:
        """
        Safely retrieves the current global memory usage in MB.
        Uses the previous measurement as a fallback on error.

        Returns:
            int: Current global memory usage in MB.
        """
        try:
            current_global_memory_bytes = psutil.virtual_memory().used
            current_global_memory_mb = int(current_global_memory_bytes / (1024 * 1024))
            logger.debug(f"[ScalingMemCheck] Current global memory usage: {current_global_memory_mb} MB")
            return current_global_memory_mb
        except Exception as e:
            logger.error(
                f"[ScalingMemCheck] Failed to get current system memory: {e}. Using previous value.", exc_info=False
            )
            # Use previous value if available, otherwise default to 0 (less ideal)
            return self.prev_global_memory_usage or 0

    def _calculate_scaling_adjustments(
        self, current_stage_metrics: Dict[str, Dict[str, Any]], current_global_memory_mb: int
    ) -> Dict[str, int]:
        """
        Runs the PID and ResourceConstraintManager to determine final replica adjustments.

        Parameters:
            current_stage_metrics: Metrics gathered by _gather_controller_metrics.
            current_global_memory_mb: Memory usage from _get_current_global_memory.

        Returns:
            Dict[str, int]: Dictionary mapping stage name to final target replica count.
        """
        logger.debug("[ScalingCalc] Calculating adjustments via PID and RCM...")
        num_edges = len(self.edge_queues)  # Get current edge count

        try:
            # 1. Get initial proposals from PID controller
            initial_proposals = self.pid_controller.calculate_initial_proposals(current_stage_metrics)
            logger.debug(
                f"[ScalingCalc] PID Initial Proposals: "
                f"{ {n: p.proposed_replicas for n, p in initial_proposals.items()} }"  # noqa
            )

            # 2. Apply constraints using Resource Constraint Manager
            # Requires initial proposals, current memory, *previous* memory, and num edges
            final_adjustments = self.constraint_manager.apply_constraints(
                initial_proposals=initial_proposals,
                current_global_memory_usage=current_global_memory_mb,
                num_edges=num_edges,
            )
            logger.debug(f"[ScalingCalc] RCM Final Adjustments: {final_adjustments}")
            return final_adjustments

        except Exception as e:
            logger.error(f"[ScalingCalc] Error during controller execution: {e}", exc_info=True)
            # Fallback: Propose no change if controllers fail
            logger.warning("[ScalingCalc] Falling back to no scaling changes due to controller error.")
            # Create a dict with current replica counts as the target
            fallback_adjustments = {}
            for stage_name, metrics in current_stage_metrics.items():
                fallback_adjustments[stage_name] = metrics.get("replicas", 0)
            return fallback_adjustments

    def _apply_scaling_actions(self, final_adjustments: Dict[str, int]) -> None:
        """
        Applies the calculated scaling adjustments concurrently using _scale_stage.

        Parameters:
            final_adjustments: Dictionary from stage name to target replica count.
        """
        scaling_futures = []
        stages_needing_action = []

        # Identify stages that require scaling
        for stage_name, target_replica_count in final_adjustments.items():
            current_count = len(self.stage_actors.get(stage_name, []))
            if target_replica_count != current_count:
                stages_needing_action.append((stage_name, target_replica_count))
                logger.info(
                    f"[ScalingApply-{stage_name}] Action required: Current={current_count},"
                    f" Target={target_replica_count}"
                )
            elif self.scaling_state.get(stage_name) != "Idle":
                # Reset state if no change needed but state wasn't Idle
                self.scaling_state[stage_name] = "Idle"

        if not stages_needing_action:
            logger.debug("[ScalingApply] No scaling actions required in this cycle.")
            return

        # Execute scaling actions concurrently
        logger.debug(f"[ScalingApply] Submitting {len(stages_needing_action)} scaling actions...")
        # Use max_workers=len(self.stages) or a smaller fixed number?
        # Using len(self.stages) might be excessive if only a few scale.
        max_workers = min(len(stages_needing_action), 8)  # Example: Limit concurrency somewhat
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for stage_name, target_count in stages_needing_action:
                # Submit the scaling task, _scale_stage handles its own logging/errors
                scaling_futures.append(executor.submit(self._scale_stage, stage_name, target_count))

        # Wait for all submitted scaling operations initiated in *this cycle* to complete
        if scaling_futures:
            logger.debug(f"[ScalingApply] Waiting for {len(scaling_futures)} scaling actions to complete...")
            # The wait ensures one scaling cycle finishes before the next evaluation starts,
            # preventing potential rapid oscillations or conflicting actions.
            done, not_done = concurrent.futures.wait(scaling_futures, timeout=120.0)  # Add timeout

            if not_done:
                logger.warning(f"[ScalingApply] Timeout waiting for {len(not_done)} scaling actions to complete.")
                # Consider logging which stages timed out if possible/needed

            # Check for exceptions in completed futures (optional, _scale_stage logs errors)
            exceptions_count = 0
            for future in done:
                if future.exception():
                    exceptions_count += 1
                    # Error already logged by _scale_stage or its helpers
            if exceptions_count > 0:
                logger.warning(
                    f"[ScalingApply] {exceptions_count} scaling actions completed with errors (see previous logs)."
                )

            logger.debug("[ScalingApply] Scaling action submission and wait completed for this cycle.")

    # --- Refactored _perform_scaling_and_maintenance Method ---
    def _perform_scaling_and_maintenance(self) -> None:
        """
        Orchestrates the scaling and maintenance cycle for the pipeline.
        """
        logger.debug("--- Performing Scaling & Maintenance Cycle ---")

        # Check 1: Skip if flushing is in progress
        if self._is_flushing:
            logger.debug("Skipping scaling cycle: Queue flush in progress.")
            return

        # Step 0: If the pipeline is quiet, and we've exceeded the interval, flush the queues
        if self._is_pipeline_quiet():
            self._execute_queue_flush()
            return

        # Step 1: Update stage statistics (processing, in_flight)
        try:
            self._update_stage_stats()
            logger.debug("Stage statistics updated.")
        except Exception as e:
            logger.error(f"Failed to update stage stats, skipping scaling cycle: {e}", exc_info=True)
            return  # Cannot proceed without fresh stats

        # Step 2: Gather metrics needed for controllers
        current_stage_metrics = self._gather_controller_metrics()

        # Step 3: Get current global memory usage
        current_global_memory_mb = self._get_current_global_memory()

        # Step 4: Calculate final scaling adjustments using controllers
        # Pass the metrics, current memory, and the *previous* cycle's memory
        final_adjustments = self._calculate_scaling_adjustments(current_stage_metrics, current_global_memory_mb)

        # Step 5: IMPORTANT - Update previous memory *after* controllers have used it
        # Store the value we just measured for the *next* cycle's RCM calculation
        self.prev_global_memory_usage = current_global_memory_mb

        # Step 6: Apply the calculated scaling actions concurrently
        self._apply_scaling_actions(final_adjustments)

        logger.debug("--- Scaling & Maintenance Cycle Complete ---")

    # --- Monitoring Thread ---
    def _get_monitor_data(self) -> List[Tuple]:
        """Helper function to fetch and format data for display."""
        # Update stats (might be redundant if scaling loop also calls it, but ensures freshness for display)
        # Consider potential race conditions if scaling modifies actors while stats are fetched.
        # Locking might be needed in a complex scenario, but keep it simple for now.
        try:
            self._update_stage_stats()
        except Exception as e:
            logger.error(f"Error updating stats in _get_monitor_data: {e}", exc_info=True)
            # Return empty data or last known good? Empty seems safer.
            return []

        output_rows = []
        for stage in self.stages:
            stage_name = stage.name
            replicas = self.stage_actors.get(stage_name, [])
            stats = self.stage_stats.get(stage_name, {"processing": 0, "in_flight": 0})

            replicas_str = f"{len(replicas)}/{stage.max_replicas}"
            if stage.min_replicas > 0:
                replicas_str += f" (min {stage.min_replicas})"

            input_edges = [ename for ename in self.edge_queues if ename.endswith(f"_to_{stage_name}")]
            occupancy_parts = []
            if input_edges:
                for ename in input_edges:
                    try:
                        q, max_q = self.edge_queues[ename]
                        # qsize might require remote call, assume _update_stage_stats handled errors
                        # We use the in_flight - processing from stats as the queue depth here
                        q_depth = stats.get("in_flight", 0) - stats.get("processing", 0)
                        occupancy_parts.append(f"{q_depth}/{max_q}")
                    except Exception:
                        occupancy_parts.append("ERR/ERR")
                occupancy_str = ", ".join(occupancy_parts) if occupancy_parts else "N/A"
            else:
                occupancy_str = "(Source)" if stage.is_source else "N/A"

            scaling_state = self.scaling_state.get(stage_name, "Unknown")
            processing_count = stats.get("processing", 0)
            stage_in_flight = stats.get("in_flight", 0)

            output_rows.append(
                (stage_name, replicas_str, occupancy_str, scaling_state, str(processing_count), str(stage_in_flight))
            )

        global_processing = sum(s.get("processing", 0) for s in self.stage_stats.values())
        global_in_flight = self._get_global_in_flight()
        output_rows.append(
            (
                "[bold]Total Pipeline[/bold]",
                "",
                "",
                f"Flushing: {self._is_flushing}",
                str(global_processing),
                str(global_in_flight),
            )
        )
        return output_rows

    def _monitor_pipeline_loop(self, poll_interval: float = 5.0) -> None:
        """
        Main loop for the monitoring thread. Handles either Rich or GUI display.
        """
        logger.debug(
            f"Monitoring thread started (Mode: {'GUI' if self.use_gui else 'Console'}, Interval: {poll_interval}s)."
        )

        if self.use_gui:
            # Attempt to create GUI instance
            self._display_instance = GuiUtilizationDisplay(refresh_rate_ms=int(poll_interval * 1000))
            if self._display_instance:
                # Start the blocking GUI loop, passing the data fetching function
                # This function will run until the GUI window is closed or stop() is called
                self._display_instance.start(self._get_monitor_data)
                logger.debug("GUI display loop finished.")
            else:
                # GUI initialization failed (e.g., no display), fall back to console
                logger.warning("GUI initialization failed. Falling back to console monitoring.")
                self.use_gui = False  # Disable GUI flag internally
                self._monitor_pipeline_loop(poll_interval)  # Retry with console mode
                return  # Exit this attempt

        else:  # Console mode (Rich)
            self._display_instance = UtilizationDisplay(refresh_rate=poll_interval)
            self._display_instance.start()  # Start the Rich Live display

            while self._monitoring:
                start_time = time.time()
                try:
                    monitor_data = self._get_monitor_data()
                    if self._display_instance and hasattr(self._display_instance, "update"):
                        self._display_instance.update(monitor_data)
                except Exception as e:
                    logger.error(f"Error in console monitoring loop: {e}", exc_info=True)

                # Sleep accounting for processing time
                elapsed = time.time() - start_time
                sleep_time = max(0, poll_interval - elapsed)
                # Use threading.Event for interruptible sleep if needed, but time.sleep is simpler
                if self._monitoring and sleep_time > 0:
                    time.sleep(sleep_time)

            # Stop Rich display if it exists
            if self._display_instance and hasattr(self._display_instance, "stop"):
                self._display_instance.stop()

        # Clean up display instance reference
        self._display_instance = None
        logger.debug("Monitoring thread loop finished.")

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

    def _scaling_loop(self, poll_interval: float = 10.0) -> None:
        logger.debug(f"Scaling/Maintenance loop started (Interval: {poll_interval}s).")
        while self._scaling_monitoring:
            start_time = time.time()
            try:
                self._perform_scaling_and_maintenance()
            except Exception as e:
                logger.error(f"Error in scaling/maintenance loop: {e}", exc_info=True)

            elapsed = time.time() - start_time
            sleep_time = max(0, poll_interval - elapsed)
            if self._scaling_monitoring and sleep_time > 0:
                logger.debug(f"Scaling loop sleeping for {sleep_time:.2f}s.")
                time.sleep(sleep_time)
        logger.debug("Scaling/Maintenance loop stopped.")

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
    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 10.0) -> None:
        """
        Start the pipeline: start actors, monitoring, and scaling.

        Parameters are the same as before.
        """
        if not self.stage_actors:
            logger.error("Cannot start pipeline: Build() must be called first.")
            return

        logger.info("Starting pipeline execution...")
        start_refs = []
        for stage_name, actors in self.stage_actors.items():
            for actor in actors:
                if hasattr(actor, "start"):
                    start_refs.append(actor.start.remote())

        if start_refs:
            logger.debug(f"Waiting for {len(start_refs)} actors to start...")
            try:
                ray.get(start_refs)
                logger.info(f"{len(start_refs)} actors started successfully.")
            except Exception as e:
                logger.error(f"Error during actor start confirmation: {e}", exc_info=True)
                self.stop()  # Attempt cleanup
                raise RuntimeError("Failed to start pipeline: error confirming actor starts.") from e

        # Start background threads
        self._start_queue_monitoring(poll_interval=monitor_poll_interval)
        self._start_scaling(poll_interval=scaling_poll_interval)
        logger.info("Pipeline started successfully.")

    def stop(self) -> None:
        """
        Stop the pipeline: stop background threads and all stage actors.
        """
        logger.info("Stopping pipeline...")

        # 1. Stop background threads first
        self._stop_scaling()
        self._stop_queue_monitoring()  # This now handles stopping Rich or GUI

        # 2. Stop actors (graceful first, then kill)
        logger.debug("Stopping all stage actors...")
        stop_refs = []
        actors_to_kill = []
        for stage_name, actors in self.stage_actors.items():
            for actor in actors:
                if hasattr(actor, "stop"):
                    stop_refs.append(actor.stop.remote())
                else:
                    actors_to_kill.append(actor)

        if stop_refs:
            logger.debug(f"Waiting for {len(stop_refs)} actors to stop gracefully...")
            try:
                # Add timeout
                ready, not_ready = ray.wait(stop_refs, num_returns=len(stop_refs), timeout=60.0)
                if not_ready:
                    logger.warning(
                        f"Timeout waiting for {len(not_ready)} actors to stop gracefully. Will proceed to kill."
                    )
                    # Add actors that timed out to the kill list
                    # This requires mapping refs back to actors - might be complex or store actor refs directly
                    # For simplicity now, we just proceed to kill those without stop()
                logger.info(f"{len(ready)} actors stopped via stop().")
            except Exception as e:
                logger.error(f"Error during actor stop confirmation: {e}", exc_info=True)

        if actors_to_kill:
            logger.debug(f"Killing {len(actors_to_kill)} actors...")
            for actor in actors_to_kill:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as e:
                    logger.error(f"Failed to kill actor {actor}: {e}")

        logger.info("Pipeline stopped.")
