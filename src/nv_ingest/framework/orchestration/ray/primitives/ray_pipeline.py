# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FunctionType

import psutil
import uuid
import ray
from ray.exceptions import GetTimeoutError
from ray.util.queue import Queue as RayQueue
from typing import Dict, Optional, List, Tuple, Any
from pydantic import BaseModel
import concurrent.futures
import logging
import time

from nv_ingest.framework.orchestration.ray.primitives.pipeline_topology import PipelineTopology, StageInfo
from nv_ingest.framework.orchestration.process.termination import kill_pipeline_process_group
from nv_ingest.framework.orchestration.ray.primitives.ray_stat_collector import RayStatsCollector
from nv_ingest.framework.orchestration.ray.util.pipeline.pid_controller import PIDController, ResourceConstraintManager
from nv_ingest.framework.orchestration.ray.util.pipeline.tools import wrap_callable_as_stage
from nv_ingest_api.util.imports.callable_signatures import ingest_stage_callable_signature
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_callable_from_path

logger = logging.getLogger(__name__)


class PipelineInterface(ABC):
    """
    Abstract base class for pipeline implementations.

    Any concrete pipeline must implement start and stop methods.
    """

    @abstractmethod
    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Start the pipeline.

        Parameters
        ----------
        monitor_poll_interval : float
            Interval in seconds for the monitoring poll (default: 5.0).
        scaling_poll_interval : float
            Interval in seconds for scaling decisions (default: 30.0).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the pipeline and perform any necessary cleanup.
        """
        pass


# --- Configuration Objects ---


@dataclass
class ScalingConfig:
    """Configuration for PID and Resource Constraint Manager based scaling."""

    dynamic_memory_scaling: bool = True
    dynamic_memory_threshold: float = 0.75
    pid_kp: float = 0.2
    pid_ki: float = 0.01
    pid_kd: float = 0.0
    pid_ema_alpha: float = 0.1
    pid_target_queue_depth: int = 0
    pid_penalty_factor: float = 0.1
    pid_error_boost_factor: float = 1.5
    rcm_memory_safety_buffer_fraction: float = 0.15


@dataclass
class FlushingConfig:
    """Configuration for queue flushing behavior."""

    queue_flush_interval_seconds: int = 600
    queue_flush_drain_timeout_seconds: int = 300
    quiet_period_threshold: int = 0
    consecutive_quiet_cycles_for_flush: int = 3


@dataclass
class StatsConfig:
    """Configuration for the RayStatsCollector."""

    collection_interval_seconds: float = 10.0
    actor_timeout_seconds: float = 5.0
    queue_timeout_seconds: float = 2.0


class RayPipelineSubprocessInterface(PipelineInterface):
    """
    Pipeline interface implementation for a subprocess-based Ray pipeline.
    """

    def __init__(self, process: multiprocessing.Process):
        """
        Parameters
        ----------
        process : multiprocessing.Process
            A handle to the running subprocess.
        """
        self._process: multiprocessing.Process = process

    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Start is not supported because the subprocess is assumed to already be running.
        """
        pass

    def stop(self) -> None:
        """
        Stops the subprocess pipeline and its entire process group to ensure
        any child processes (e.g., the simple message broker) are terminated.
        """
        try:
            pid = int(self._process.pid)
        except Exception:
            return

        # Always attempt to terminate the entire process group
        try:
            kill_pipeline_process_group(pid)
        except Exception as e:
            logger.warning(f"kill_pipeline_process_group failed: {e}")


class RayPipelineInterface(PipelineInterface):
    """
    Pipeline interface for an in-process RayPipeline instance.
    """

    def __init__(self, pipeline: "RayPipeline"):
        """
        Parameters
        ----------
        pipeline : RayPipeline
            The instantiated pipeline to control.
        """
        self._pipeline = pipeline

    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Starts the RayPipeline.

        Parameters
        ----------
        monitor_poll_interval : float
            Unused here; provided for interface compatibility.
        scaling_poll_interval : float
            Unused here; provided for interface compatibility.
        """
        self._pipeline.start(monitor_poll_interval, scaling_poll_interval)

    def stop(self) -> None:
        """
        Stops the RayPipeline and shuts down Ray.
        """
        self._pipeline.stop()

        try:
            import ray

            ray.shutdown()
        except Exception:
            pass


class RayPipeline(PipelineInterface):
    """
    A structured pipeline supporting dynamic scaling and queue flushing.
    Uses PIDController and ResourceConstraintManager. Supports optional GUI display.
    Delegates statistics collection to RayStatsCollector.

    Configuration is managed via dedicated config objects (ScalingConfig, etc.).
    """

    def __init__(
        self,
        scaling_config: ScalingConfig = ScalingConfig(),
        flushing_config: FlushingConfig = FlushingConfig(),
        stats_config: StatsConfig = StatsConfig(),
    ) -> None:
        """
        Initializes the RayPipeline.

        Parameters
        ----------
        scaling_config : ScalingConfig, optional
            Configuration for PID and resource constraint-based scaling, by default ScalingConfig().
        flushing_config : FlushingConfig, optional
            Configuration for queue flushing behavior, by default FlushingConfig().
        stats_config : StatsConfig, optional
            Configuration for the RayStatsCollector, by default StatsConfig().
        """
        # Store config objects
        self.scaling_config = scaling_config
        self.flushing_config = flushing_config
        self.stats_config = stats_config

        # --- Instantiate Topology ---
        self.topology = PipelineTopology()

        # --- Structure Lock ---
        self._structure_lock: threading.Lock = threading.Lock()

        # --- State ---
        # self.scaling_state: Dict[str, str] = {}
        self.prev_global_memory_usage: Optional[int] = None
        self._state_lock: threading.Lock = threading.Lock()
        self._stopping = False

        # --- Build Time Config & State ---
        # Use scaling_config for these
        self.dynamic_memory_scaling = self.scaling_config.dynamic_memory_scaling
        self.dynamic_memory_threshold = self.scaling_config.dynamic_memory_threshold

        # --- Background Threads ---
        self._scaling_thread: Optional[threading.Thread] = None
        self._scaling_monitoring = False

        # --- Queue Flushing ---
        self._last_queue_flush_time: float = time.time()
        self.queue_flush_interval_seconds = self.flushing_config.queue_flush_interval_seconds
        self.queue_flush_drain_timeout_seconds = self.flushing_config.queue_flush_drain_timeout_seconds
        self.quiet_period_threshold = self.flushing_config.quiet_period_threshold
        self.consecutive_quiet_cycles_for_flush = self.flushing_config.consecutive_quiet_cycles_for_flush
        self._consecutive_quiet_cycles = 0

        # --- Instantiate Autoscaling Controllers ---
        # Use scaling_config
        self.pid_controller = PIDController(
            kp=self.scaling_config.pid_kp,
            ki=self.scaling_config.pid_ki,
            kd=self.scaling_config.pid_kd,
            target_queue_depth=self.scaling_config.pid_target_queue_depth,
            penalty_factor=self.scaling_config.pid_penalty_factor,
            error_boost_factor=self.scaling_config.pid_error_boost_factor,
        )
        logger.debug("PIDController initialized using ScalingConfig.")

        try:
            total_system_memory_bytes = psutil.virtual_memory().total
            # Use scaling_config for dynamic_memory_threshold
            absolute_memory_threshold_mb = int(
                self.scaling_config.dynamic_memory_threshold * total_system_memory_bytes / (1024 * 1024)
            )
        except Exception as e:
            logger.error(f"Failed to get system memory: {e}. Using high limit.")
            absolute_memory_threshold_mb = 1_000_000  # Fallback value

        # Use scaling_config
        self.constraint_manager = ResourceConstraintManager(
            max_replicas=1,  # Updated during build
            memory_threshold=absolute_memory_threshold_mb,
            memory_safety_buffer_fraction=self.scaling_config.rcm_memory_safety_buffer_fraction,
        )
        logger.debug("ResourceConstraintManager initialized using ScalingConfig.")

        # --- Instantiate Stats Collector ---
        self._stats_collection_interval_seconds = self.stats_config.collection_interval_seconds
        self.stats_collector = RayStatsCollector(
            pipeline_accessor=self,  # This dependency remains for now
            interval=self.stats_config.collection_interval_seconds,
            actor_timeout=self.stats_config.actor_timeout_seconds,
            queue_timeout=self.stats_config.queue_timeout_seconds,
            ema_alpha=self.scaling_config.pid_ema_alpha,
        )

        logger.debug("RayStatsCollector initialized using StatsConfig.")

    # --- Accessor Methods for Stat Collector (and internal use) ---

    def __del__(self):
        """Ensures the pipeline is stopped upon garbage collection."""
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Exception during RayPipeline cleanup: {e}")

    def get_stages_info(self) -> List[StageInfo]:
        """
        Returns a snapshot of the current stage information.

        Returns
        -------
        List[StageInfo]
            A list of StageInfo objects from the topology.
        """
        return self.topology.get_stages_info()

    def get_stage_actors(self) -> Dict[str, List[Any]]:
        """
        Returns a snapshot of the current actors per stage.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping stage names to lists of actor handles.
        """
        return self.topology.get_stage_actors()

    def get_edge_queues(self) -> Dict[str, Tuple[Any, int]]:
        """
        Returns a snapshot of the current edge queues.

        Returns
        -------
        Dict[str, Tuple[Any, int]]
            A dictionary mapping queue names to tuples of (queue_handle, queue_size).
        """
        return self.topology.get_edge_queues()

    def _configure_autoscalers(self) -> None:
        """Updates controllers based on current pipeline configuration via topology."""
        logger.debug("[Build-Configure] Configuring autoscalers...")
        total_max_replicas = 0
        default_cost_bytes = 100 * 1024 * 1024
        stage_overheads = {}  # Collect locally

        # Use topology accessor
        current_stages = self.topology.get_stages_info()

        for stage in current_stages:
            total_max_replicas += stage.max_replicas
            # Use estimated overhead if available (Assume it's calculated elsewhere or default)
            # For now, let's store a dummy overhead in topology during build
            overhead_bytes = default_cost_bytes  # Simplification for now
            stage_overheads[stage.name] = overhead_bytes  # Store locally first

        # Update topology with collected overheads
        self.topology.set_stage_memory_overhead(stage_overheads)

        # Update constraint manager
        self.constraint_manager.max_replicas = total_max_replicas

        logger.debug(f"[Build-Configure] Autoscalers configured. Total Max Replicas: {total_max_replicas}")

    def _instantiate_initial_actors(self) -> None:
        """Instantiates initial actors and updates topology."""
        logger.debug("[Build-Actors] Instantiating initial stage actors (min_replicas)...")
        # Use topology accessor
        current_stages = self.topology.get_stages_info()

        for stage in current_stages:
            replicas = []

            if not self.dynamic_memory_scaling:
                num_initial_actors = stage.max_replicas
            else:
                num_initial_actors = (
                    max(stage.min_replicas, 1) if stage.is_source or stage.is_sink else stage.min_replicas
                )

            if num_initial_actors > 0:
                logger.debug(f"[Build-Actors] Stage '{stage.name}' creating {num_initial_actors} initial actor(s).")
                for i in range(num_initial_actors):
                    actor_name = f"{stage.name}_{uuid.uuid4()}"
                    logger.debug(
                        f"[Build-Actors] Creating actor '{actor_name}' ({i + 1}/{num_initial_actors})"
                        f" for '{stage.name}'"
                    )
                    try:
                        actor = stage.callable.options(name=actor_name, max_concurrency=1, max_restarts=0).remote(
                            config=stage.config, stage_name=stage.name
                        )
                        replicas.append(actor)
                    except Exception as e:
                        logger.error(f"[Build-Actors] Failed create actor '{actor_name}': {e}", exc_info=True)
                        raise RuntimeError(f"Build failed: actor creation error for stage '{stage.name}'") from e

            # Update topology for this stage
            self.topology.set_actors_for_stage(stage.name, replicas)
            logger.debug(f"[Build-Actors] Stage '{stage.name}' initial actors set in topology: count={len(replicas)}")

        logger.debug("[Build-Actors] Initial actor instantiation complete.")

    def _create_and_wire_edges(self) -> List[ray.ObjectRef]:
        """
        Creates queues, wires actors (using topology), and updates topology.

        Returns
        -------
        List[ray.ObjectRef]
            A list of object references for the remote wiring calls.
        """
        logger.debug("[Build-Wiring] Creating and wiring edges...")
        wiring_refs = []
        new_edge_queues: Dict[str, Tuple[Any, int]] = {}

        current_connections = self.topology.get_connections()
        current_stage_actors = self.topology.get_stage_actors()  # Gets copy

        for from_stage_name, connections_list in current_connections.items():
            for to_stage_name, queue_size in connections_list:
                queue_name = f"{from_stage_name}_to_{to_stage_name}"
                logger.debug(f"[Build-Wiring] Creating queue '{queue_name}' (size {queue_size}) and wiring.")
                try:
                    edge_queue = RayQueue(maxsize=queue_size, actor_options={"max_restarts": 0})
                    new_edge_queues[queue_name] = (edge_queue, queue_size)

                    # Wire using current actors from topology snapshot
                    source_actors = current_stage_actors.get(from_stage_name, [])
                    for actor in source_actors:
                        wiring_refs.append(actor.set_output_queue.remote(edge_queue))

                    dest_actors = current_stage_actors.get(to_stage_name, [])
                    for actor in dest_actors:
                        wiring_refs.append(actor.set_input_queue.remote(edge_queue))

                except Exception as e:
                    logger.error(f"[Build-Wiring] Failed create/wire queue '{queue_name}': {e}", exc_info=True)
                    raise RuntimeError(f"Build failed: queue wiring error for '{queue_name}'") from e

        # Update topology with the new queues
        self.topology.set_edge_queues(new_edge_queues)

        logger.debug(f"[Build-Wiring] Submitted {len(wiring_refs)} wiring calls. Queues set in topology.")
        return wiring_refs

    @staticmethod
    def _wait_for_wiring(wiring_refs: List[ray.ObjectRef]) -> None:
        """
        Waits for remote wiring calls to complete.

        Parameters
        ----------
        wiring_refs : List[ray.ObjectRef]
            A list of object references for the wiring calls.
        """
        if not wiring_refs:
            logger.debug("[Build-WaitWiring] No wiring calls.")
            return
        logger.debug(f"[Build-WaitWiring] Waiting for {len(wiring_refs)} wiring calls...")
        try:
            ray.get(wiring_refs)
            logger.debug("[Build-WaitWiring] All wiring calls completed.")
        except Exception as e:
            logger.error(f"[Build-WaitWiring] Error during wiring confirmation: {e}", exc_info=True)
            raise RuntimeError("Build failed: error confirming initial wiring") from e

    def add_source(
        self, *, name: str, source_actor: Any, config: BaseModel, min_replicas: int = 1, max_replicas: int = 1
    ) -> "RayPipeline":
        """
        Adds a source stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the source stage.
        source_actor : Any
            The actor or callable for the source stage.
        config : BaseModel
            The configuration for the source stage.
        min_replicas : int, optional
            The minimum number of replicas for the source stage, by default 1.
        max_replicas : int, optional
            The maximum number of replicas for the source stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance.
        """
        if min_replicas < 1:
            logger.warning(f"Source stage '{name}': min_replicas must be >= 1. Overriding.")
            min_replicas = 1

        stage_info = StageInfo(
            name=name,
            callable=source_actor,
            config=config,
            is_source=True,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.topology.add_stage(stage_info)  # Delegate

        return self

    def add_stage(
        self,
        *,
        name: str,
        stage_actor: Any,
        config: BaseModel,
        min_replicas: int = 0,
        max_replicas: int = 1,
    ) -> "RayPipeline":
        """
        Adds a stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the stage.
        stage_actor : Any
            The actor or callable for the stage.
        config : BaseModel
            The configuration for the stage.
        min_replicas : int, optional
            The minimum number of replicas for the stage, by default 0.
        max_replicas : int, optional
            The maximum number of replicas for the stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance.
        """
        if min_replicas < 0:
            logger.warning(f"Stage '{name}': min_replicas cannot be negative. Overriding to 0.")
            min_replicas = 0

        resolved_actor = stage_actor

        # Support module path (e.g., "mypkg.mymodule:my_lambda")
        if isinstance(stage_actor, str):
            resolved_actor = resolve_callable_from_path(
                callable_path=stage_actor, signature_schema=ingest_stage_callable_signature
            )

        # Wrap callables
        if isinstance(resolved_actor, FunctionType):
            schema_type = type(config)
            resolved_actor = wrap_callable_as_stage(resolved_actor, schema_type)

        stage_info = StageInfo(
            name=name,
            callable=resolved_actor,
            config=config,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.topology.add_stage(stage_info)

        return self

    def add_sink(
        self, *, name: str, sink_actor: Any, config: BaseModel, min_replicas: int = 1, max_replicas: int = 1
    ) -> "RayPipeline":
        """
        Adds a sink stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the sink stage.
        sink_actor : Any
            The actor or callable for the sink stage.
        config : BaseModel
            The configuration for the sink stage.
        min_replicas : int, optional
            The minimum number of replicas for the sink stage, by default 1.
        max_replicas : int, optional
            The maximum number of replicas for the sink stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance.
        """
        # Sink min_replicas can realistically be 0 if data drain is optional/best-effort? Let's allow 0.
        if min_replicas < 0:
            logger.warning(f"Sink stage '{name}': min_replicas cannot be negative. Overriding to 0.")
            min_replicas = 0
        stage_info = StageInfo(
            name=name,
            callable=sink_actor,
            config=config,
            is_sink=True,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.topology.add_stage(stage_info)  # Delegate

        return self

    # --- Method for defining connections ---
    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        """
        Creates an edge between two stages in the pipeline.

        Parameters
        ----------
        from_stage : str
            The name of the source stage.
        to_stage : str
            The name of the destination stage.
        queue_size : int, optional
            The size of the queue between the stages, by default 100.

        Returns
        -------
        RayPipeline
            The pipeline instance.
        """
        try:
            self.topology.add_connection(from_stage, to_stage, queue_size)  # Delegate (includes validation)
        except ValueError as e:
            logger.error(f"make_edge failed: {e}")
            raise  # Re-raise the error
        return self

    # ----- Pipeline Build Process ---
    def build(self) -> Dict[str, List[Any]]:
        """
        Builds the pipeline: configures, instantiates, wires, using topology.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping stage names to lists of actor handles.
        """
        logger.debug("--- Starting Pipeline Build Process ---")
        try:
            if not self.topology.get_stages_info():
                logger.error("Build failed: No stages defined in topology.")
                return {}

            # Steps interact with self.topology
            self._configure_autoscalers()
            self._instantiate_initial_actors()
            wiring_futures = self._create_and_wire_edges()
            self._wait_for_wiring(wiring_futures)

            logger.debug("--- Pipeline Build Completed Successfully ---")
            return self.topology.get_stage_actors()  # Return actors from topology

        except RuntimeError as e:
            logger.critical(f"Pipeline build failed: {e}", exc_info=False)
            # Clean up topology runtime state?
            self.topology.clear_runtime_state()
            return {}
        except Exception as e:
            logger.critical(f"Unexpected error during pipeline build: {e}", exc_info=True)
            self.topology.clear_runtime_state()
            return {}

    # --- Scaling Logic ---
    @staticmethod
    def _create_single_replica(stage_info: StageInfo) -> Any:
        """
        Creates a single new Ray actor replica for the given stage.

        Parameters
        ----------
        stage_info : StageInfo
            The stage information.

        Returns
        -------
        Any
            The new actor handle.
        """
        actor_name = f"{stage_info.name}_{uuid.uuid4()}"
        logger.debug(f"[ScaleUtil] Creating new actor '{actor_name}' for stage '{stage_info.name}'")
        try:
            new_actor = stage_info.callable.options(name=actor_name, max_concurrency=1, max_restarts=0).remote(
                config=stage_info.config, stage_name=stage_info.name
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
        """
        Gets wiring futures for a single actor using topology for queues/connections.

        Parameters
        ----------
        actor : Any
            The actor handle.
        stage_name : str
            The name of the stage.

        Returns
        -------
        List[ray.ObjectRef]
            A list of object references for the wiring calls.
        """
        wiring_refs = []

        # Use topology accessors
        connections = self.topology.get_connections()
        edge_queues = self.topology.get_edge_queues()

        # Wire outputs
        if stage_name in connections:
            for to_stage, _ in connections[stage_name]:
                queue_name = f"{stage_name}_to_{to_stage}"
                if queue_name in edge_queues:
                    edge_queue, _ = edge_queues[queue_name]
                    wiring_refs.append(actor.set_output_queue.remote(edge_queue))

        # Wire inputs
        for from_stage, conns in connections.items():
            for to_stage, _ in conns:
                if to_stage == stage_name:
                    queue_name = f"{from_stage}_to_{stage_name}"
                    if queue_name in edge_queues:
                        edge_queue, _ = edge_queues[queue_name]
                        wiring_refs.append(actor.set_input_queue.remote(edge_queue))

        return wiring_refs

    @staticmethod
    def _start_actors(actors_to_start: List[Any], stage_name: str) -> None:
        """
        Starts a list of actors if they have a 'start' method and waits for completion.

        Parameters
        ----------
        actors_to_start : List[Any]
            A list of actor handles.
        stage_name : str
            The name of the stage.
        """
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
        """
        Handles scaling up, interacting with topology.

        Parameters
        ----------
        stage_info : StageInfo
            The stage information.
        current_count : int
            The current number of replicas.
        target_count : int
            The target number of replicas.
        """
        stage_name = stage_info.name
        num_to_add = target_count - current_count
        logger.debug(f"[ScaleUp-{stage_name}] Scaling up from {current_count} to {target_count} (+{num_to_add}).")
        # Update topology state
        self.topology.update_scaling_state(stage_name, "Scaling Up")

        new_actors = []
        all_wiring_refs = []
        successfully_added_actors = []

        try:
            # 1. Create actors
            for _ in range(num_to_add):
                new_actor = self._create_single_replica(stage_info)
                new_actors.append(new_actor)

            # 2. Get wiring refs (uses topology internally)
            for actor in new_actors:
                all_wiring_refs.extend(self._get_wiring_refs_for_actor(actor, stage_name))

            # 3. Wait for wiring (static helper)
            self._wait_for_wiring(all_wiring_refs)  # Handles errors

            # 4. Start actors (static helper)
            self._start_actors(new_actors, stage_name)  # Handles errors

            # 5. Add successfully created/wired/started actors to topology
            for actor in new_actors:
                self.topology.add_actor_to_stage(stage_name, actor)
                successfully_added_actors.append(actor)  # Keep track

            final_count = self.topology.get_actor_count(stage_name)
            logger.debug(
                f"[ScaleUp-{stage_name}] Scale up complete. Added {len(successfully_added_actors)}. "
                f"New count: {final_count}"
            )

        except Exception as e:
            logger.error(f"[ScaleUp-{stage_name}] Error during scale up: {e}", exc_info=False)
            self.topology.update_scaling_state(stage_name, "Error")
            # --- Cleanup Attempt ---
            # Actors created but potentially not wired/started/added to topology.
            # Only kill actors that were definitely *not* added to the topology.
            actors_to_kill = [a for a in new_actors if a not in successfully_added_actors]
            if actors_to_kill:
                logger.warning(
                    f"[ScaleUp-{stage_name}] Attempting to kill {len(actors_to_kill)} partially created actors."
                )
                for actor in actors_to_kill:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception as kill_e:
                        logger.warning(f"Failed to kill actor {actor}: {kill_e}")
            logger.critical(f"[ScaleUp-{stage_name}] Scale up failed. State potentially inconsistent.")

        finally:
            # Reset state only if it was Scaling Up and didn't end in Error
            current_state = self.topology.get_scaling_state().get(stage_name)
            if current_state == "Scaling Up":
                self.topology.update_scaling_state(stage_name, "Idle")

    def _handle_scale_down(self, stage_name: str, current_replicas: List[Any], target_count: int) -> None:
        """
        Handles scaling down: initiates stop on actors and marks them for removal
        by the topology's garbage collection mechanism.

        Parameters
        ----------
        stage_name : str
            The name of the stage.
        current_replicas : List[Any]
            A list of actor handles.
        target_count : int
            The target number of replicas.
        """
        current_count = len(current_replicas)
        num_to_remove = current_count - target_count
        logger.debug(
            f"[ScaleDown-{stage_name}] Scaling down from {current_count} to {target_count} (-{num_to_remove})."
        )

        if num_to_remove <= 0:
            return

        # Select actors to remove (e.g., the most recently added)
        actors_to_remove = current_replicas[-num_to_remove:]

        logger.debug(f"[ScaleDown-{stage_name}] Selected {len(actors_to_remove)} actors for removal.")

        # Signal each actor to stop and mark it for removal by the topology.
        # The topology's cleanup thread will handle polling and final removal.
        for actor in actors_to_remove:
            try:
                actor.stop.remote()  # Signal the actor's loop to stop
                self.topology.mark_actor_for_removal(stage_name, actor)
            except Exception as e:
                logger.error(f"[ScaleDown-{stage_name}] Failed to initiate stop for actor {actor}: {e}")

        logger.debug(f"[ScaleDown-{stage_name}] Scale down initiation complete for {len(actors_to_remove)} actors.")

    def _scale_stage(self, stage_name: str, new_replica_count: int) -> None:
        """
        Orchestrates scaling using topology for state and info.

        Parameters
        ----------
        stage_name : str
            The name of the stage.
        new_replica_count : int
            The new number of replicas.
        """
        logger.debug(f"[ScaleStage-{stage_name}] Request for target count: {new_replica_count}")

        # --- Use Topology Accessors ---
        stage_info = self.topology.get_stage_info(stage_name)
        current_replicas = self.topology.get_stage_actors().get(stage_name, [])  # Get current actors safely
        current_count = len(current_replicas)

        if stage_info is None:
            logger.error(f"[ScaleStage-{stage_name}] Stage info not found. Cannot scale.")
            return

        target_count = max(stage_info.min_replicas, min(new_replica_count, stage_info.max_replicas))
        if target_count != new_replica_count:
            logger.debug(
                f"[ScaleStage-{stage_name}] Count {new_replica_count} adjusted to {target_count} "
                f"by bounds ({stage_info.min_replicas}/{stage_info.max_replicas})."
            )

        if target_count == current_count:
            logger.debug(f"[ScaleStage-{stage_name}] Already at target count ({current_count}). No action.")
            # Reset state if needed
            if self.topology.get_scaling_state().get(stage_name) != "Idle":
                self.topology.update_scaling_state(stage_name, "Idle")
            return

        # --- Delegate ---
        try:
            if target_count > current_count:
                self._handle_scale_up(stage_info, current_count, target_count)
            else:  # target_count < current_count
                # Pass the list of actors we know about *now*
                self._handle_scale_down(stage_name, current_replicas, target_count)
        except RuntimeError as e:  # Catch specific errors from handlers
            logger.error(f"[ScaleStage-{stage_name}] Scaling failed: {e}", exc_info=False)
            # State should have been set to "Error" within the handler
        except Exception as e:
            logger.error(f"[ScaleStage-{stage_name}] Unexpected error: {e}", exc_info=True)
            self.topology.update_scaling_state(stage_name, "Error")  # Ensure error state

    def _is_pipeline_quiet(self) -> bool:
        """
        Checks if pipeline is quiet using topology state and stats collector.

        Returns
        -------
        bool
            True if the pipeline is quiet, False otherwise.
        """
        return False  # TODO: disabled for debugging

        # Check topology state first
        if self.topology.get_is_flushing():
            logger.debug("Pipeline quiet check: False (Flush in progress via topology state)")
            return False

        # Check stats collector for recent activity
        latest_stats = self.stats_collector.get_latest_stats()
        if not latest_stats:
            logger.debug("Pipeline quiet check: True (No stats available, assuming quiet)")
            return True  # No stats could mean it's idle

        total_in_flight = latest_stats.get("total_items_in_flight", 0)
        logger.debug(f"Pipeline quiet check: Total in-flight items: {total_in_flight}")

        return total_in_flight <= self.quiet_period_threshold

    def _wait_for_pipeline_drain(self, timeout_seconds: int) -> bool:
        """
        Actively monitors pipeline drain using direct calls to the stats collector.

        Parameters
        ----------
        timeout_seconds : int
            The timeout in seconds.

        Returns
        -------
        bool
            True if the pipeline drained successfully, False otherwise.
        """
        start_time = time.time()
        logger.debug(f"Waiting for pipeline drain (Timeout: {timeout_seconds}s)...")
        last_in_flight = -1
        drain_check_interval = 1.0  # Check every second

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= timeout_seconds:
                logger.warning(f"Pipeline drain timed out after {elapsed_time:.1f}s. Last In-Flight: {last_in_flight}")
                return False

            # --- Trigger immediate stats collection via the collector instance ---
            drain_success = False
            collection_error = None

            global_in_flight = -1
            try:
                # Use the collector's method for a one-off, blocking collection
                drain_stats, global_in_flight, drain_success = self.stats_collector.collect_stats_now()
            except Exception as e:
                logger.error(f"[DrainWait] Critical error during direct stats collection call: {e}.", exc_info=True)
                collection_error = e  # Indicate failure to even run collection

            # --- Process collection results ---
            if global_in_flight != last_in_flight:
                status_msg = (
                    f"Collection Success: {drain_success}"
                    if not collection_error
                    else f"Collection Error: {type(collection_error).__name__}"
                )
                logger.debug(
                    f"[Drain] Check at {elapsed_time:.1f}s: Global In-Flight={global_in_flight} ({status_msg})"
                )
                last_in_flight = global_in_flight

            # --- Check for successful drain ---
            # Requires BOTH in-flight=0 AND the collection reporting it was successful
            if global_in_flight == 0 and drain_success and not collection_error:
                return True
            elif global_in_flight == 0:  # Saw zero, but collection wasn't fully successful
                logger.warning(
                    "[Drain] In-Flight reached 0, but stats collection had errors/timeouts."
                    " Cannot confirm drain yet."
                )

            # --- Wait ---
            remaining_time = timeout_seconds - elapsed_time
            sleep_duration = min(drain_check_interval, remaining_time, 1.0)  # Ensure positive sleep
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    def _execute_queue_flush(self) -> bool:
        """
        Executes queue flush, using topology for state and structure.

        Returns
        -------
        bool
            True if the flush was successful, False otherwise.
        """
        if self.topology.get_is_flushing() or self._stopping:  # Check topology state
            logger.warning("Queue flush requested but already in progress or pipeline is stopping. Ignoring.")
            return False

        # Set flushing state in topology
        self.topology.set_flushing(True)
        overall_success = False
        source_actors_paused = []
        pause_refs = []
        new_edge_queues_map: Optional[Dict[str, Tuple[Any, int]]] = None

        try:
            # --- Get structure snapshots from topology ---
            # Use lock context for multiple reads if needed, but individual accessors are locked too
            current_stages = self.topology.get_stages_info()
            current_stage_actors = self.topology.get_stage_actors()
            current_edge_queues = self.topology.get_edge_queues()
            current_connections = self.topology.get_connections()

            # --- 1. Pause Source Stages (using snapshots) ---
            logger.debug("Pausing source stages...")
            pause_timeout = 60.0
            for stage in current_stages:
                if stage.is_source:
                    actors = current_stage_actors.get(stage.name, [])
                    for actor in actors:
                        if hasattr(actor, "pause") and hasattr(actor.pause, "remote"):
                            try:
                                pause_refs.append(actor.pause.remote())
                                source_actors_paused.append(actor)
                            except Exception as e:
                                logger.error(f"Failed sending pause to {actor}: {e}")

            if pause_refs:
                logger.debug(f"Waiting up to {pause_timeout}s for {len(pause_refs)} sources to pause...")
                try:
                    ray.get(pause_refs, timeout=pause_timeout)
                    logger.debug(f"{len(pause_refs)} sources acknowledged pause.")
                except GetTimeoutError:
                    logger.warning(f"Timeout waiting for {len(pause_refs)} sources to pause.")
                except Exception as e:
                    logger.error(f"Error waiting for sources pause: {e}. Proceeding cautiously.")

            # --- 2. Wait for Drain ---
            logger.debug("Waiting for pipeline to drain...")
            if not self._wait_for_pipeline_drain(self.queue_flush_drain_timeout_seconds):
                raise RuntimeError("Pipeline drain failed or timed out, aborting flush.")

            # --- 3. Create New Queues (using snapshot) ---
            logger.debug("Creating new replacement queues...")
            new_edge_queues_map = {}
            for queue_name, (_, queue_size) in current_edge_queues.items():
                try:
                    new_edge_queues_map[queue_name] = (
                        RayQueue(maxsize=queue_size, actor_options={"max_restarts": 0}),
                        queue_size,
                    )
                    logger.debug(f"Created new queue: {queue_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to create new queue '{queue_name}'.") from e

            # --- 4. Re-wire Actors to New Queues (using snapshots) ---
            logger.debug("Re-wiring actors to new queues...")
            wiring_refs = []
            wiring_timeout = 120.0
            for from_stage_name, conns in current_connections.items():
                for to_stage_name, _ in conns:
                    queue_name = f"{from_stage_name}_to_{to_stage_name}"
                    if queue_name not in new_edge_queues_map:
                        raise RuntimeError(f"New queue missing for {queue_name}")
                    new_queue_actor, _ = new_edge_queues_map[queue_name]

                    # Re-wire sources outputs
                    for actor in current_stage_actors.get(from_stage_name, []):
                        try:
                            wiring_refs.append(actor.set_output_queue.remote(new_queue_actor))
                        except Exception as e:
                            logger.error(f"Failed sending set_output_queue to {actor}: {e}")

                    # Re-wire destinations inputs
                    for actor in current_stage_actors.get(to_stage_name, []):
                        try:
                            wiring_refs.append(actor.set_input_queue.remote(new_queue_actor))
                        except Exception as e:
                            logger.error(f"Failed sending set_input_queue to {actor}: {e}")

            if wiring_refs:
                logger.debug(f"Waiting up to {wiring_timeout}s for {len(wiring_refs)} actors to re-wire...")
                try:
                    ready, not_ready = ray.wait(wiring_refs, num_returns=len(wiring_refs), timeout=wiring_timeout)
                    if not_ready:
                        raise RuntimeError("Actor re-wiring timed out or failed.")
                    ray.get(ready)  # Check for internal errors
                    logger.debug(f"{len(ready)} actors re-wired successfully.")
                except Exception as e:
                    raise RuntimeError("Actor re-wiring failed.") from e

            # --- 5. Update Topology State (Commit Point) ---
            logger.debug("Committing new queues to pipeline topology.")
            self.topology.set_edge_queues(new_edge_queues_map)  # Commit the change
            overall_success = True

        except Exception as e:
            logger.error(f"Error during queue flush: {e}", exc_info=True)
            overall_success = False

        finally:
            # --- 6. Resume Source Stages (Always attempt) ---
            if source_actors_paused:
                logger.debug(f"Attempting to resume {len(source_actors_paused)} source actors...")
                resume_timeout = 30.0
                resume_refs = []
                for actor in source_actors_paused:
                    try:
                        resume_refs.append(actor.resume.remote())
                    except Exception as e:
                        logger.error(f"Failed sending resume to {actor}: {e}")
                if resume_refs:
                    logger.debug(f"Waiting up to {resume_timeout}s for {len(resume_refs)} actors to resume...")
                    try:
                        ray.get(resume_refs, timeout=resume_timeout)
                        logger.debug(f"{len(resume_refs)} sources resumed.")
                    except GetTimeoutError:
                        logger.warning(f"Timeout waiting for {len(resume_refs)} sources to resume.")
                    except Exception as e:
                        logger.error(f"Error waiting for sources resume: {e}")

            # Update flush timestamp only on success
            if overall_success:
                self._last_queue_flush_time = time.time()

            # Reset flushing state in topology
            self.topology.set_flushing(False)

        return overall_success

    def request_queue_flush(self, force: bool = False) -> None:
        """
        Requests a queue flush, checking topology state.

        Parameters
        ----------
        force : bool, optional
            Whether to force the flush, by default False.
        """
        logger.debug(f"Manual queue flush requested (force={force}).")

        if self.topology.get_is_flushing() or self._stopping:  # Check topology
            logger.warning("Flush already in progress or pipeline is stopping.")
            return
        if force or self._is_pipeline_quiet():
            # Consider running _execute_queue_flush in a separate thread
            # to avoid blocking the caller, especially if 'force=True'.
            # For now, run synchronously:
            self._execute_queue_flush()
        else:
            logger.debug("Manual flush denied: pipeline not quiet or interval not met.")

    def _gather_controller_metrics(
        self, current_stage_stats: Dict[str, Dict[str, int]], global_in_flight: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Gathers metrics using provided stats and topology.

        Parameters
        ----------
        current_stage_stats : Dict[str, Dict[str, int]]
            The current stage statistics.
        global_in_flight : int
            The global in-flight count.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary of metrics for the controllers.
        """
        logger.debug("[ScalingMetrics] Gathering metrics for controllers...")
        current_stage_metrics = {}

        # Use topology accessors
        current_stages = self.topology.get_stages_info()
        current_actors = self.topology.get_stage_actors()  # Snapshot

        for stage in current_stages:
            stage_name = stage.name
            replicas = len(current_actors.get(stage_name, []))
            stats = current_stage_stats.get(stage_name, {"processing": 0, "in_flight": 0})
            processing = stats.get("processing", 0)
            in_flight = stats.get("in_flight", 0)
            queue_depth = max(0, in_flight - processing)

            current_stage_metrics[stage_name] = {
                "replicas": replicas,
                "queue_depth": queue_depth,
                "processing": processing,
                "in_flight": in_flight,
                "min_replicas": stage.min_replicas,
                "max_replicas": stage.max_replicas,
                "pipeline_in_flight": global_in_flight,
            }

        logger.debug(f"[ScalingMetrics] Gathered metrics for {len(current_stage_metrics)} stages.")
        return current_stage_metrics

    def _get_current_global_memory(self) -> int:
        """
        Safely retrieves the current global system memory usage (used, not free) in MB.

        Returns
        -------
        int
            The current global memory usage (RSS/used) in MB.
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
        self, current_stage_metrics: Dict[str, Dict[str, Any]], global_in_flight: int, current_global_memory_mb: int
    ) -> Dict[str, int]:
        """
        Runs controllers to get target replica counts using topology for edge count.

        Parameters
        ----------
        current_stage_metrics : Dict[str, Dict[str, Any]]
            The current stage metrics.
        global_in_flight : int
            The global in-flight count.
        current_global_memory_mb : int
            The current global memory usage in MB.

        Returns
        -------
        Dict[str, int]
            A dictionary of target replica counts for the stages.
        """
        logger.debug("[ScalingCalc] Calculating adjustments via PID and RCM...")
        # Get edge count from topology
        num_edges = len(self.topology.get_edge_queues())

        try:
            initial_proposals = self.pid_controller.calculate_initial_proposals(current_stage_metrics)
            logger.debug(
                "[ScalingCalc] PID Initial Proposals:"
                f" { {n: p.proposed_replicas for n, p in initial_proposals.items()} }"  # noqa E201,E202
            )

            final_adjustments = self.constraint_manager.apply_constraints(
                initial_proposals=initial_proposals,
                global_in_flight=global_in_flight,
                current_global_memory_usage_mb=current_global_memory_mb,
                num_edges=num_edges,
            )
            logger.debug(f"[ScalingCalc] RCM Final Adjustments: {final_adjustments}")
            return final_adjustments
        except Exception as e:
            logger.error(f"[ScalingCalc] Error during controller execution: {e}", exc_info=True)
            logger.warning("[ScalingCalc] Falling back to current replica counts.")
            return {name: metrics.get("replicas", 0) for name, metrics in current_stage_metrics.items()}

    def _apply_scaling_actions(self, final_adjustments: Dict[str, int]) -> None:
        """
        Applies scaling by calling _scale_stage, using topology for validation.

        Parameters
        ----------
        final_adjustments : Dict[str, int]
            A dictionary of target replica counts for the stages.
        """
        stages_needing_action = []
        current_actors_map = self.topology.get_stage_actors()  # Snapshot

        for stage_name, target_replica_count in final_adjustments.items():
            current_count = len(current_actors_map.get(stage_name, []))
            stage_info = self.topology.get_stage_info(stage_name)  # Get info from topology

            if not stage_info:
                logger.warning(f"[ScalingApply] Cannot apply scaling for unknown stage '{stage_name}'. Skipping.")
                continue

            # Clamp target using StageInfo from topology
            clamped_target = max(stage_info.min_replicas, min(stage_info.max_replicas, target_replica_count))
            if clamped_target != target_replica_count:
                logger.warning(
                    f"[ScalingApply-{stage_name}] Target {target_replica_count} clamped to {clamped_target} by bounds."
                )
                target_replica_count = clamped_target

            if target_replica_count != current_count:
                stages_needing_action.append((stage_name, target_replica_count))
                logger.debug(
                    f"[ScalingApply-{stage_name}] Action: Current={current_count}, "
                    f"Target={target_replica_count} (Min={stage_info.min_replicas}, Max={stage_info.max_replicas})"
                )

        if not stages_needing_action:
            logger.debug("[ScalingApply] No scaling actions required.")
            return

        max_workers = min(len(stages_needing_action), 8)
        logger.debug(
            f"[ScalingApply] Submitting {len(stages_needing_action)} scaling actions ({max_workers} workers)..."
        )
        action_results = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ScalingAction"
        ) as executor:
            future_to_stage = {
                executor.submit(self._scale_stage, stage_name, target_count): stage_name
                for stage_name, target_count in stages_needing_action
            }
            wait_timeout = 180.0
            logger.debug(f"[ScalingApply] Waiting up to {wait_timeout}s for actions...")
            for future in concurrent.futures.as_completed(future_to_stage, timeout=wait_timeout):
                stage_name = future_to_stage[future]
                try:
                    result = future.result()  # Raises exception if _scale_stage failed internally
                    action_results[stage_name] = {"status": "completed", "result": result}
                    logger.debug(f"[ScalingApply-{stage_name}] Action completed.")
                except TimeoutError:
                    logger.error(f"[ScalingApply-{stage_name}] Action timed out ({wait_timeout}s).")
                    action_results[stage_name] = {"status": "timeout"}
                    self.topology.update_scaling_state(stage_name, "Error")  # Mark as error on timeout
                except Exception as exc:
                    logger.error(f"[ScalingApply-{stage_name}] Action failed: {exc}", exc_info=True)
                    action_results[stage_name] = {"status": "error", "exception": exc}
                    # State should be set to Error inside _scale_stage or its handlers on failure

        completed = sum(1 for r in action_results.values() if r["status"] == "completed")
        errors = sum(1 for r in action_results.values() if r["status"] == "error")
        timeouts = sum(1 for r in action_results.values() if r["status"] == "timeout")
        logger.debug(f"[ScalingApply] Summary: {completed} completed, {errors} errors, {timeouts} timeouts.")

    def _perform_scaling_and_maintenance(self) -> None:
        """
        Orchestrates scaling/maintenance using topology and stats collector.
        """
        if self._stopping:
            logger.debug("Pipeline is stopping. Skipping scaling cycle.")
            return

        if not self.dynamic_memory_scaling:
            logger.debug("Dynamic memory scaling disabled. Skipping cycle.")
            return

        if self.topology.get_is_flushing():
            logger.debug("Skipping scaling cycle: Queue flush in progress (topology state).")
            return

        got_lock = self._state_lock.acquire(timeout=0.1)
        if not got_lock:
            logger.debug("Could not acquire lock for maintenance; skipping cycle.")
            return

        cycle_start_time = time.time()
        try:
            if self._stopping:
                logger.debug("Pipeline began stopping after acquiring lock. Skipping maintenance logic.")
                return

            logger.debug("--- Performing Scaling & Maintenance Cycle ---")

            if self._is_pipeline_quiet():
                self._consecutive_quiet_cycles += 1
                logger.debug(f"Pipeline is quiet. Consecutive quiet cycles: {self._consecutive_quiet_cycles}")
                if self._consecutive_quiet_cycles >= self.consecutive_quiet_cycles_for_flush:
                    logger.debug(
                        f"Pipeline has been quiet for {self._consecutive_quiet_cycles} cycles. "
                        "Initiating queue flush."
                    )
                    if self._execute_queue_flush():
                        self._last_queue_flush_time = time.time()
                    self._consecutive_quiet_cycles = 0  # Reset after attempting flush
                else:
                    logger.debug(
                        f"Pipeline is quiet, but waiting for {self.consecutive_quiet_cycles_for_flush} "
                        "consecutive quiet cycles before flushing."
                    )
            else:
                if self._consecutive_quiet_cycles > 0:
                    logger.debug(
                        f"Pipeline is no longer quiet. Resetting consecutive quiet cycle count "
                        f"from {self._consecutive_quiet_cycles} to 0."
                    )
                self._consecutive_quiet_cycles = 0
                logger.debug("Queue flush interval reached, but pipeline is not quiet. Deferring.")

            # Fast return check if stopping occurred while flushing or checking flush status
            if self._stopping:
                return

            current_stage_stats, global_in_flight, last_update_time, stats_were_successful = (
                self.stats_collector.get_latest_stats()
            )

            last_update_age = time.time() - last_update_time
            max_age = max(15.0, self._stats_collection_interval_seconds)
            if not current_stage_stats or not stats_were_successful or last_update_age > max_age:
                status = "No stats" if not current_stage_stats else "Failed" if not stats_were_successful else "Stale"
                logger.warning(
                    f"[Scaling] Cannot scale reliably: Stats {status} (Age: {last_update_age:.1f}s). Skipping cycle."
                )
                return

            current_stage_metrics = self._gather_controller_metrics(current_stage_stats, global_in_flight)
            if not current_stage_metrics:
                logger.error("[Scaling] Failed to gather metrics. Skipping.")
                return

            current_global_memory_mb = self._get_current_global_memory()
            final_adjustments = self._calculate_scaling_adjustments(
                current_stage_metrics, global_in_flight, current_global_memory_mb
            )
            self.prev_global_memory_usage = current_global_memory_mb
            self._apply_scaling_actions(final_adjustments)

            logger.debug(
                f"--- Scaling & Maintenance Cycle Complete (Duration: {time.time() - cycle_start_time:.2f}s) ---"
            )

        except Exception as e:  # noqa
            logger.error("Exception during maintenance cycle", exc_info=True)

        finally:
            self._state_lock.release()

    # --- Lifecycle Methods for Monitoring/Scaling Threads ---
    def _scaling_loop(self, interval: float) -> None:
        """
        Main loop for the scaling thread.

        Parameters
        ----------
        interval : float
            The interval in seconds.
        """
        logger.debug(f"Scaling loop started. Interval: {interval}s")
        while self._scaling_monitoring:
            try:
                self._perform_scaling_and_maintenance()
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}", exc_info=True)

            sleep_time = interval
            if not self._scaling_monitoring:
                break
            time.sleep(sleep_time)
        logger.debug("Scaling loop finished.")

    def _start_scaling(self, poll_interval: float = 10.0) -> None:
        """
        Starts the scaling thread.

        Parameters
        ----------
        poll_interval : float, optional
            The interval in seconds, by default 10.0.
        """
        if not self._scaling_monitoring:
            self._scaling_monitoring = True
            self._scaling_thread = threading.Thread(target=self._scaling_loop, args=(poll_interval,), daemon=True)
            self._scaling_thread.start()
            logger.debug(f"Scaling/Maintenance thread launched (Interval: {poll_interval}s).")

    def _stop_scaling(self) -> None:
        """
        Stops the scaling thread.
        """
        if self._scaling_monitoring:
            logger.debug("Stopping scaling/maintenance thread...")
            self._scaling_monitoring = False
            if self._scaling_thread is not None:
                self._scaling_thread.join(timeout=15)  # Allow more time for scaling actions
                if self._scaling_thread.is_alive():
                    logger.warning("Scaling thread did not exit cleanly.")
            self._scaling_thread = None
            logger.debug("Scaling/Maintenance stopped.")

    # --- Pipeline Start/Stop ---
    def start(self, monitor_poll_interval: float = 5.0, scaling_poll_interval: float = 30.0) -> None:
        """
        Starts the pipeline actors and background monitoring threads.

        Assumes the pipeline has already been built via the `build()` method.

        Parameters
        ----------
        monitor_poll_interval : float, optional
            This parameter is currently unused but is kept for interface compatibility.
        scaling_poll_interval : float, optional
            The interval in seconds for the autoscaling and maintenance thread to run, by default 30.0.
        """
        if not self.topology.get_all_actors():
            logger.error("Cannot start: Pipeline not built or has no actors.")
            return

        logger.info("Starting pipeline execution...")

        # Start all actors
        actors_to_start = self.topology.get_all_actors()
        start_futures = [actor.start.remote() for actor in actors_to_start]
        if start_futures:
            logger.debug(f"Waiting for {len(start_futures)} actors to start...")
            try:
                ray.get(start_futures, timeout=60.0)
                logger.debug(f"{len(start_futures)} actors started.")
            except Exception as e:
                logger.error(f"Error/Timeout starting actors: {e}", exc_info=True)
                self.stop()  # Attempt cleanup
                raise RuntimeError("Pipeline start failed: actors did not start.") from e

        # Start background threads
        self.topology.start_cleanup_thread()
        self.stats_collector.start()
        self._start_scaling(poll_interval=scaling_poll_interval)

        logger.info("Pipeline started successfully.")

    def stop(self) -> None:
        """
        Stops the pipeline and all associated actors and threads.

        This method performs a graceful shutdown by:
        1.  Stopping the autoscaling and statistics collection threads.
        2.  Signaling all actors to stop and waiting for confirmation.
        3.  Stopping the topology cleanup thread.
        4.  Clearing all runtime state from the topology.
        """
        logger.info("Stopping pipeline...")

        if self._stopping:
            logger.warning("Stop already in progress.")
            return
        self._stopping = True

        # 1. Stop background threads first to prevent new actions
        self._stop_scaling()
        self.stats_collector.stop()

        # 2. Stop all actors and wait for them
        logger.debug("Stopping all actors in the topology...")
        all_actors = self.topology.get_all_actors()
        if all_actors:
            stop_futures = [actor.stop.remote() for actor in all_actors]
            try:
                ready, not_ready = ray.wait(stop_futures, num_returns=len(stop_futures), timeout=60.0)
                if not_ready:
                    logger.warning(
                        f"Timeout waiting for {len(not_ready)} actors to stop. " f"Proceeding with shutdown."
                    )
                logger.debug(f"{len(ready)} actors confirmed stop.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during actor shutdown: {e}", exc_info=True)

        # 3. Stop the topology cleanup thread now that actors are stopped
        self.topology.stop_cleanup_thread()

        # 4. Clear all state and references
        logger.debug("Clearing pipeline topology runtime state.")
        self.topology.clear_runtime_state()

        self._stopping = False
        logger.info("Pipeline stopped successfully.")

    def __enter__(self) -> "RayPipeline":
        """
        Enter the runtime context related to this object.

        Returns
        -------
        RayPipeline
            The pipeline instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the runtime context related to this object.
        """
        self.stop()
