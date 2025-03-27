# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import concurrent
import threading
import time
import uuid  # For generating unique actor names
from dataclasses import dataclass
from typing import Any, Dict, List

import psutil
import ray
from pydantic import BaseModel
import logging

from rich.console import Console
from rich.table import Table
from rich.live import Live

from ray.util.queue import Queue

from nv_ingest.framework.orchestration.ray.util.pipeline.pid_controller import PIDController
from nv_ingest.framework.orchestration.ray.util.system_tools.memory import estimate_actor_memory_overhead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    """
    Information about a pipeline stage.
    """

    name: str
    callable: Any  # A Ray remote actor class
    config: BaseModel
    is_source: bool = False
    is_sink: bool = False
    min_replicas: int = 0
    max_replicas: int = 1


class RayPipeline:
    """
    A structured pipeline that supports source, intermediate, and sink stages.
    Stages are connected using Ray distributed queues directly.
    """

    def __init__(
        self,
        scaling_threshold: float = 10.0,
        scaling_cooldown: int = 20,
        dynamic_memory_scaling: bool = False,
        dynamic_memory_threshold: float = 0.9,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.0,
    ) -> None:
        """
        Initialize the RayPipeline instance.

        Parameters
        ----------
        scaling_threshold : float, optional
            Threshold for scaling decisions (as a percentage), by default 10.0.
        scaling_cooldown : int, optional
            Number of cycles to wait before scaling down, by default 20.
        dynamic_memory_scaling : bool, optional
            Flag indicating whether dynamic memory scaling is enabled, by default False.
        dynamic_memory_threshold : float, optional
            Fraction of system memory to use as the threshold for scaling, by default 0.9.

        Attributes
        ----------
        stages : list of StageInfo
            List of stages added to the pipeline.
        connections : dict
            Mapping from source stage names to lists of tuples (to_stage, queue_size).
        stage_actors : dict
            Mapping from stage names to lists of actor handles.
        edge_queues : dict
            Mapping from edge names to (queue, max_size) pairs.
        queue_stats : dict
            Mapping from edge names to lists of queue utilization statistics.
        under_threshold_cycles : dict
            Mapping from stage names to cycle counts under threshold.
        dynamic_memory_scaling : bool
            Whether dynamic memory scaling is enabled.
        dynamic_memory_threshold : float
            Memory threshold for dynamic scaling (in bytes).
        stage_memory_overhead : dict
            Estimated memory overhead per stage.
        idle : bool
            Flag indicating if the pipeline is in an idle state.
        scaling_state : dict
            Current scaling state per stage.
        _monitoring : bool
            Flag indicating if queue monitoring is active.
        _monitor_thread : threading.Thread
            Thread used for queue monitoring.
        _scaling_monitoring : bool
            Flag indicating if scaling monitoring is active.
        _scaling_thread : threading.Thread
            Thread used for scaling.
        """

        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        # Instead of edge actors, we'll store (queue, max_size)
        self.edge_queues: Dict[str, Any] = {}
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}
        self.scaling_threshold: float = scaling_threshold
        self.scaling_cooldown: int = scaling_cooldown
        self.under_threshold_cycles: Dict[str, int] = {}  # stage name -> cycle count

        self.dynamic_memory_scaling = dynamic_memory_scaling
        self.dynamic_memory_threshold = dynamic_memory_threshold  # In bytes
        self.stage_memory_overhead: Dict[str, float] = {}

        self.idle: bool = True
        self.scaling_state: Dict[str, str] = {}

        self._monitoring: bool = False
        self._monitor_thread: threading.Thread = None
        self._scaling_monitoring: bool = False
        self._scaling_thread: threading.Thread = None

        # Create a PIDController instance.
        total_system_memory = psutil.virtual_memory().total
        memory_threshold_mb = int(dynamic_memory_threshold * total_system_memory / (1024 * 1024))
        # For now we initialize stage_cost_estimates as empty; it will be populated in build().
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            max_replicas=100,  # This will be updated later based on the stages.
            memory_threshold=memory_threshold_mb,
            stage_cost_estimates={},
        )
        logger.info(
            "RayPipeline initialized with PID scaling: kp=%s, ki=%s, kd=%s, memory_threshold=%s MB",
            pid_kp,
            pid_ki,
            pid_kd,
            memory_threshold_mb,
        )

        logger.info(
            "RayPipeline initialized with idle=True, dynamic_memory_scaling=%s, dynamic_memory_threshold=%s",
            self.dynamic_memory_scaling,
            self.dynamic_memory_threshold,
        )

    def add_source(
        self, *, name: str, source_actor: Any, config: BaseModel, min_replicas: int = 1, max_replicas: int = 1
    ) -> "RayPipeline":
        """
        Add a source stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the source stage.
        source_actor : Any
            The Ray remote actor class for the source stage.
        config : BaseModel
            Configuration for the source stage.
        min_replicas : int, optional
            Minimum number of replicas for the source stage, by default 1.
        max_replicas : int, optional
            Maximum number of replicas for the source stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance with the source stage added.
        """
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
        """
        Add an intermediate stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the intermediate stage.
        stage_actor : Any
            The Ray remote actor class for the stage.
        config : BaseModel
            Configuration for the stage.
        min_replicas : int, optional
            Minimum number of replicas for the stage, by default 0.
        max_replicas : int, optional
            Maximum number of replicas for the stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance with the intermediate stage added.
        """
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
        """
        Add a sink stage to the pipeline.

        Parameters
        ----------
        name : str
            The name of the sink stage.
        sink_actor : Any
            The Ray remote actor class for the sink stage.
        config : BaseModel
            Configuration for the sink stage.
        min_replicas : int, optional
            Minimum number of replicas for the sink stage, by default 1.
        max_replicas : int, optional
            Maximum number of replicas for the sink stage, by default 1.

        Returns
        -------
        RayPipeline
            The pipeline instance with the sink stage added.
        """
        if min_replicas < 1:
            logger.warning(f"Sink stage '{name}': min_replicas must be at least 1. Overriding to 1.")
            min_replicas = 1
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

    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        """
        Create an edge (connection) between two stages in the pipeline.

        Parameters
        ----------
        from_stage : str
            The name of the source stage.
        to_stage : str
            The name of the destination stage.
        queue_size : int, optional
            The maximum size of the distributed queue for the edge, by default 100.

        Returns
        -------
        RayPipeline
            The pipeline instance with the new edge added.

        Raises
        ------
        ValueError
            If either the from_stage or to_stage is not found in the pipeline.
        """
        if from_stage not in [s.name for s in self.stages]:
            logger.error(f"make_edge: Stage {from_stage} not found")
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            logger.error(f"make_edge: Stage {to_stage} not found")
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        logger.debug(f"Created edge from {from_stage} to {to_stage} with queue_size {queue_size}")
        return self

    def build(self) -> Dict[str, List[Any]]:
        logger.debug("Building pipeline: Instantiating stage actors...")

        if self.dynamic_memory_scaling:
            logger.info("Dynamic memory scaling enabled. Estimating per-stage memory overhead...")
            total_overhead = 0.0
            for stage in self.stages:
                logger.info("Estimating overhead for stage '%s'...", stage.name)
                overhead = estimate_actor_memory_overhead(
                    stage.callable, actor_kwargs={"config": stage.config, "progress_engine_count": -1}
                )
                self.stage_memory_overhead[stage.name] = overhead
                total_overhead += overhead
                logger.info("Stage '%s' overhead: %.2f MB", stage.name, overhead / (1024 * 1024))
            avg_overhead = total_overhead / len(self.stages) if self.stages else 0.0
            logger.info("Average overhead per stage: %.2f MB", avg_overhead / (1024 * 1024))
            total_system_memory = psutil.virtual_memory().total
            threshold_bytes = self.dynamic_memory_threshold * total_system_memory
            logger.info(
                "Total system memory: %.2f MB; dynamic threshold: %.2f MB (%.0f%%)",
                total_system_memory / (1024 * 1024),
                threshold_bytes / (1024 * 1024),
                self.dynamic_memory_threshold * 100,
            )
            required_total_replicas = int(threshold_bytes / avg_overhead) if avg_overhead > 0 else 0
            current_total_replicas = sum(stage.max_replicas for stage in self.stages)
            logger.info(
                "Current total max replicas: %d; required replicas to meet threshold: %d",
                current_total_replicas,
                required_total_replicas,
            )
            if required_total_replicas and current_total_replicas != required_total_replicas:
                ratio = required_total_replicas / current_total_replicas
                logger.info("Adjusting max_replicas by scaling factor: %.2f", ratio)
                for stage in self.stages:
                    original = stage.max_replicas
                    stage.max_replicas = max(stage.min_replicas, int(stage.max_replicas * ratio))
                    logger.info(
                        "Stage '%s': max_replicas adjusted from %d to %d", stage.name, original, stage.max_replicas
                    )
            else:
                logger.info("No adjustment needed: current max replicas align with memory threshold.")

        # Update PID controller’s stage_cost_estimates and overall max_replicas based on the stages.
        total_max_replicas = 0
        for stage in self.stages:
            total_max_replicas += stage.max_replicas
            # Use the estimated overhead (in MB) or default to a value (e.g., 100 MB).
            overhead = self.stage_memory_overhead.get(stage.name, 100 * 1024 * 1024)
            self.pid_controller.stage_cost_estimates[stage.name] = int(overhead / (1024 * 1024))
            self.pid_controller.initialize_stage(stage.name)
        self.pid_controller.max_replicas = total_max_replicas

        # Instantiate stage actors.
        for stage in self.stages:
            replicas = []
            for _ in range(stage.min_replicas):
                actor_name = f"{stage.name}_{uuid.uuid4()}"
                logger.debug(f"Creating actor {actor_name} for stage {stage.name}")
                actor = stage.callable.options(name=actor_name, max_concurrency=100).remote(
                    config=stage.config, progress_engine_count=-1
                )
                replicas.append(actor)
            self.stage_actors[stage.name] = replicas
            self.under_threshold_cycles[stage.name] = 0
            logger.debug(f"Stage {stage.name} actors: {replicas}")

        # Wire up edges between stages.
        wiring_refs = []
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                logger.debug(f"Creating distributed queue {queue_name} with queue_size {queue_size}")
                edge_queue = Queue(maxsize=queue_size)
                self.edge_queues[queue_name] = (edge_queue, queue_size)
                self.queue_stats[queue_name] = []
                for actor in self.stage_actors.get(from_stage, []):
                    logger.debug(f"Wiring output queue for actor {actor} in stage {from_stage} to queue {queue_name}")
                    wiring_refs.append(actor.set_output_queue.remote(edge_queue))
                for actor in self.stage_actors.get(to_stage, []):
                    logger.debug(f"Wiring input queue for actor {actor} in stage {to_stage} to queue {queue_name}")
                    wiring_refs.append(actor.set_input_queue.remote(edge_queue))
        logger.debug("Waiting for all wiring calls to complete...")
        ray.get(wiring_refs)
        logger.debug("Pipeline build complete.")
        return self.stage_actors

    def _scale_stage(self, stage_name: str, new_replica_count: int) -> None:
        """
        Dynamically scale the specified stage to a new replica count.

        Parameters
        ----------
        stage_name : str
            The name of the stage to scale.
        new_replica_count : int
            The target number of replicas for the stage.
        """
        current_replicas = self.stage_actors.get(stage_name, [])
        current_count = len(current_replicas)
        stage_info = next((s for s in self.stages if s.name == stage_name), None)
        if stage_info is None:
            logger.error(f"[Scale Stage] Stage info for '{stage_name}' not found during scaling.")
            return

        if new_replica_count > current_count:
            if current_count >= stage_info.max_replicas:
                logger.debug(
                    f"[Scale Stage] Stage '{stage_name}' already at max replicas ({current_count}). Skipping scale-up."
                )
                return
            target_count = min(new_replica_count, stage_info.max_replicas)
            logger.debug(
                f"[Scale Stage] Scaling UP stage '{stage_name}' from {current_count} to {target_count} replicas."
            )
            if self.dynamic_memory_scaling:
                stage_overhead = self.stage_memory_overhead.get(stage_name, 0)
                additional_replicas = target_count - current_count
                predicted_extra = additional_replicas * stage_overhead
                total_system_memory = psutil.virtual_memory().total
                safe_limit = 0.9 * total_system_memory
                current_used = psutil.virtual_memory().used
                logger.info(
                    f"[Scale Stage] Stage '{stage_name}' predicted extra memory:"
                    f" {predicted_extra / (1024 * 1024):.2f} MB. "
                    f"Current usage: {current_used / (1024 * 1024):.2f} MB;"
                    f" Safe limit: {safe_limit / (1024 * 1024):.2f} MB."
                )
                if current_used + predicted_extra > safe_limit:
                    logger.warning(
                        f"[Scale Stage] Scaling up stage '{stage_name}'"
                        f" would exceed 90% of system memory. Aborting scale-up."
                    )
                    return

            for _ in range(target_count - current_count):
                actor_name = f"{stage_name}_{uuid.uuid4()}"
                logger.debug(f"[Scale Stage] Creating new replica '{actor_name}' for stage '{stage_name}'")
                new_actor = stage_info.callable.options(name=actor_name, max_concurrency=10).remote(
                    config=stage_info.config, progress_engine_count=-1
                )
                wiring_refs = []
                if stage_name in self.connections:
                    for to_stage, _ in self.connections[stage_name]:
                        queue_name = f"{stage_name}_to_{to_stage}"
                        if queue_name in self.edge_queues:
                            edge_queue, _ = self.edge_queues[queue_name]
                            logger.debug(
                                f"[Scale Stage] Wiring new actor '{actor_name}' output to queue '{queue_name}'"
                            )
                            wiring_refs.append(new_actor.set_output_queue.remote(edge_queue))
                        else:
                            logger.error(
                                f"[Scale Stage] Output queue '{queue_name}' not found for stage '{stage_name}'"
                            )
                for from_stage, conns in self.connections.items():
                    for to_stage, _ in conns:
                        if to_stage == stage_name:
                            queue_name = f"{from_stage}_to_{stage_name}"
                            if queue_name in self.edge_queues:
                                edge_queue, _ = self.edge_queues[queue_name]
                                logger.debug(
                                    f"[Scale Stage] Wiring new actor '{actor_name}' input to queue '{queue_name}'"
                                )
                                wiring_refs.append(new_actor.set_input_queue.remote(edge_queue))
                            else:
                                logger.error(
                                    f"[Scale Stage] Input queue '{queue_name}' not found for stage '{stage_name}'"
                                )
                if wiring_refs:
                    ray.get(wiring_refs)
                    logger.debug(f"[Scale Stage] Wiring complete for new actor '{actor_name}'")
                try:
                    if hasattr(new_actor, "start"):
                        logger.debug(f"[Scale Stage] Starting new actor '{actor_name}' for stage '{stage_name}'")
                        ray.get(new_actor.start.remote())
                    else:
                        logger.warning(
                            f"[Scale Stage] New actor '{actor_name}' for stage '{stage_name}' has no start() method."
                        )
                except Exception as e:
                    logger.error(f"[Scale Stage] Error starting new actor '{actor_name}': {e}")
                current_replicas.append(new_actor)
                logger.debug(f"[Scale Stage] New replica '{actor_name}' created, wired, and started.")
            self.stage_actors[stage_name] = current_replicas
            self.scaling_state[stage_name] = "Scaling Up"
            logger.debug(
                f"[Scale Stage] Scaling UP complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )
        elif new_replica_count < current_count:
            logger.debug(
                f"[Scale Stage] Scaling DOWN stage '{stage_name}' from {current_count} to {new_replica_count} replicas."
            )
            remove_count = current_count - new_replica_count
            stopped_actors = []
            stop_futures = []
            for _ in range(remove_count):
                actor = current_replicas.pop()
                stopped_actors.append(actor)
                logger.debug(f"[Scale Stage] Stopping replica '{actor}' for stage '{stage_name}'")
                stop_futures.append(actor.stop.remote())
            try:
                ray.get(stop_futures)
                logger.debug(f"[Scale Stage] Successfully stopped {remove_count} replicas for stage '{stage_name}'")
            except Exception as e:
                logger.error(f"[Scale Stage] Error stopping actors during scaling down: {e}")
            for actor in stopped_actors:
                try:
                    ray.kill(actor, no_restart=True)
                    logger.debug(f"[Scale Stage] Successfully killed replica '{actor}' for stage '{stage_name}'")
                except Exception as e:
                    logger.error(f"[Scale Stage] Error killing actor '{actor}': {e}")
            self.stage_actors[stage_name] = current_replicas
            self.scaling_state[stage_name] = "Scaling Down"
            logger.debug(
                f"[Scale Stage] Scaling DOWN complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )

    def _idle_scale_all_stages(self) -> None:
        """
        Perform idle scaling by creating missing replicas for every stage, ensuring that
        each stage has replicas up to an idle target (capped by the maximum replicas).
        """
        logger.debug("[Idle Scale] Starting idle scale-up for all stages.")
        new_actor_dict = {}
        for stage in self.stages:
            stage_name = stage.name
            current_replicas = self.stage_actors.get(stage_name, [])
            current_count = len(current_replicas)
            if current_count >= stage.max_replicas:
                new_actor_dict[stage_name] = []
                continue
            target_count = min(max(stage.min_replicas, stage.max_replicas // 2), stage.max_replicas)
            logger.debug(
                f"[Idle Scale] Stage '{stage_name}': current_count = {current_count}, target_count = {target_count}"
            )
            new_replicas = []
            if target_count > current_count:
                for _ in range(current_count, target_count):
                    actor_name = f"{stage_name}_{uuid.uuid4()}"
                    logger.debug(f"[Idle Scale] Creating new replica '{actor_name}' for stage '{stage_name}'")
                    new_actor = stage.callable.options(name=actor_name, max_concurrency=10).remote(
                        config=stage.config, progress_engine_count=-1
                    )
                    new_replicas.append(new_actor)
            new_actor_dict[stage_name] = new_replicas

        for stage in self.stages:
            stage_name = stage.name
            self.stage_actors[stage_name] = self.stage_actors.get(stage_name, []) + new_actor_dict.get(stage_name, [])
            logger.debug(f"[Idle Scale] Stage '{stage_name}' new replica count: {len(self.stage_actors[stage_name])}")

        wiring_refs = []
        for stage in self.stages:
            stage_name = stage.name
            for new_actor in new_actor_dict.get(stage_name, []):
                if stage_name in self.connections:
                    for to_stage, _ in self.connections[stage_name]:
                        queue_name = f"{stage_name}_to_{to_stage}"
                        if queue_name in self.edge_queues:
                            edge_queue, _ = self.edge_queues[queue_name]
                            logger.debug(f"[Idle Scale] Wiring new actor '{new_actor}' output to queue '{queue_name}'")
                            wiring_refs.append(new_actor.set_output_queue.remote(edge_queue))
                        else:
                            logger.error(f"[Idle Scale] Output queue '{queue_name}' not found for stage '{stage_name}'")
                for from_stage, conns in self.connections.items():
                    for to_stage, _ in conns:
                        if to_stage == stage_name:
                            queue_name = f"{from_stage}_to_{stage_name}"
                            if queue_name in self.edge_queues:
                                edge_queue, _ = self.edge_queues[queue_name]
                                logger.debug(
                                    f"[Idle Scale] Wiring new actor '{new_actor}' input to queue '{queue_name}'"
                                )
                                wiring_refs.append(new_actor.set_input_queue.remote(edge_queue))
                            else:
                                logger.error(
                                    f"[Idle Scale] Input queue '{queue_name}' not found for stage '{stage_name}'"
                                )
        if wiring_refs:
            ray.get(wiring_refs)
            logger.debug("[Idle Scale] Wiring complete for all new actors.")

        start_refs = []
        for stage in self.stages:
            for new_actor in new_actor_dict.get(stage.name, []):
                try:
                    if hasattr(new_actor, "start"):
                        logger.debug(f"[Idle Scale] Starting new actor '{new_actor}' for stage '{stage.name}'")
                        start_refs.append(new_actor.start.remote())
                    else:
                        logger.warning(
                            f"[Idle Scale] New actor '{new_actor}' for stage '{stage.name}' has no start() method."
                        )
                except Exception as e:
                    logger.error(f"[Idle Scale] Error starting new actor '{new_actor}' for stage '{stage.name}': {e}")
        if start_refs:
            ray.get(start_refs)
            logger.debug("[Idle Scale] All new actors started.")

        for s in self.stages:
            self.under_threshold_cycles[s.name] = 0
            self.scaling_state[s.name] = "Idle Scale Up"
        logger.debug("[Idle Scale] Idle scale-up complete; all stages scaled to target replica counts.")

    def _perform_scaling(self) -> None:
        """
        Use the PID controller to evaluate current metrics for each stage and
        adjust the replica counts accordingly. This updated implementation gathers
        scaling adjustments for all stages, then concurrently executes the scaling operations,
        and finally waits until all scaling requests complete.
        """
        stage_metrics: Dict[str, Dict[str, Any]] = {}

        for stage in self.stages:
            stage_name = stage.name

            # Compute queue_depth from all input edges.
            input_edges = [edge for edge in self.edge_queues.keys() if edge.endswith(f"_to_{stage_name}")]
            queue_depth = sum(self.edge_queues[edge][0].qsize() for edge in input_edges) if input_edges else 0

            # Throughput is not measured in this example; set to 0.
            throughput = 0

            replicas = len(self.stage_actors.get(stage_name, []))

            # Gather memory usage from each actor (if available).
            memory_usage = 0
            for actor in self.stage_actors.get(stage_name, []):
                try:
                    stats = ray.get(actor.get_stats.remote())
                    mem = stats.get("memory_usage", 0)
                    memory_usage += mem
                except Exception as e:
                    logger.error(f"Error fetching stats for actor {actor} in stage {stage_name}: {e}")

            stage_metrics[stage_name] = {
                "queue_depth": queue_depth,
                "throughput": throughput,
                "replicas": replicas,
                "memory_usage": 5_000,  # Hard-coded for this example; adjust as needed.
                "target_queue_depth": 0,  # This can be tuned per stage if needed.
            }

        # Get global memory usage in MB.
        current_memory_usage_bytes = psutil.virtual_memory().used
        current_memory_usage_mb = current_memory_usage_bytes / (1024 * 1024)

        # Ask the PID controller for new replica counts.
        adjustments = self.pid_controller.update(stage_metrics, current_memory_usage_mb)

        # Launch scaling operations concurrently for all stages needing adjustment.
        scaling_futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for stage_name, new_replica_count in adjustments.items():
                current_count = len(self.stage_actors.get(stage_name, []))
                if new_replica_count != current_count:
                    logger.info(
                        f"PID scaling adjustment for stage {stage_name}: from {current_count} to {new_replica_count}"
                    )
                    scaling_futures.append(executor.submit(self._scale_stage, stage_name, new_replica_count))
                else:
                    logger.debug(
                        f"No scaling action needed for stage {stage_name}: replica count remains {current_count}"
                    )
            # Wait for all scaling operations to complete.
            concurrent.futures.wait(scaling_futures)

    def _monitor_queue_utilization(self, poll_interval: float = 10.0) -> None:
        """
        Monitor the utilization of each distributed queue and update the display with
        real-time statistics.

        Parameters
        ----------
        poll_interval : float, optional
            The time interval (in seconds) between successive polls, by default 10.0.
        """
        display = UtilizationDisplay(refresh_rate=2)
        logger.debug("Queue monitoring thread started.")
        while self._monitoring:
            current_time = time.time()
            for edge_name, (edge_queue, max_size) in self.edge_queues.items():
                current_size = edge_queue.qsize()
                utilization = (current_size / max_size) * 100 if max_size > 0 else 0
                logger.debug(f"[Monitor] Queue '{edge_name}': {current_size}/{max_size} ({utilization:.1f}%) utilized.")
                self.queue_stats[edge_name].append({"timestamp": current_time, "utilization": utilization})

            output_rows = []
            global_processing = 0
            global_queued = 0
            for stage in self.stages:
                stage_name = stage.name
                current_replicas = self.stage_actors.get(stage_name, [])
                replicas_str = f"{len(current_replicas)}/{stage.max_replicas}"
                input_edges = [
                    edge_name for edge_name in self.edge_queues.keys() if edge_name.endswith(f"_to_{stage_name}")
                ]
                occupancy_list = []
                total_queued = 0
                for edge_name in input_edges:
                    stats_list = self.queue_stats.get(edge_name, [])
                    if stats_list:
                        last_util = stats_list[-1]["utilization"]
                        _, max_size = self.edge_queues[edge_name]
                        occupancy = int((last_util / 100) * max_size)
                        total_queued += occupancy
                        occupancy_list.append(f"{occupancy}/{max_size}")
                    else:
                        occupancy_list.append("0/0")
                occupancy_str = ", ".join(occupancy_list) if occupancy_list else "N/A"

                processing_count = 0
                if current_replicas:
                    actor_futures = [actor.get_stats.remote() for actor in current_replicas]
                    actor_stats = ray.get(actor_futures)
                    for stat in actor_stats:
                        processing_count += int(stat.get("active_processing", False))
                stage_in_flight = processing_count + total_queued
                global_processing += processing_count
                global_queued += total_queued

                scaling_state = self.scaling_state.get(stage_name, "---")
                output_rows.append(
                    (
                        stage_name,
                        replicas_str,
                        occupancy_str,
                        scaling_state,
                        str(processing_count),
                        str(stage_in_flight),
                    )
                )

            global_total = global_processing + global_queued
            output_rows.append(("[bold]Total Pipeline[/bold]", "", "", "", str(global_processing), str(global_total)))
            display.update(output_rows)
            time.sleep(poll_interval)

        display.stop()
        logger.debug("Queue monitoring thread stopped.")

    def _start_queue_monitoring(self) -> None:
        """
        Start the thread responsible for monitoring queue utilization.
        """
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_queue_utilization, daemon=True)
            self._monitor_thread.start()
            logger.debug("Queue monitoring thread launched.")

    def _stop_queue_monitoring(self) -> None:
        """
        Stop the queue monitoring thread and wait for its termination.
        """
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread is not None:
                self._monitor_thread.join()
            logger.debug("Queue monitoring thread stopped.")

    def _scaling_loop(self, poll_interval: float = 10.0) -> None:
        logger.debug("Scaling thread started.")
        while self._scaling_monitoring:
            self._perform_scaling()
            time.sleep(poll_interval)
        logger.debug("Scaling thread stopped.")

    def _start_scaling(self) -> None:
        """
        Run a continuous loop that periodically evaluates scaling conditions and adjusts
        the number of stage replicas accordingly.

        Parameters
        ----------
        poll_interval : float, optional
            The time interval (in seconds) between scaling evaluations, by default 10.0.
        """
        if not self._scaling_monitoring:
            self._scaling_monitoring = True
            self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self._scaling_thread.start()
            logger.debug("Scaling thread launched.")

    def _stop_scaling(self) -> None:
        """
        Stop the scaling thread and wait for it to finish.
        """
        if self._scaling_monitoring:
            self._scaling_monitoring = False
            if self._scaling_thread is not None:
                self._scaling_thread.join()
            logger.debug("Scaling thread stopped.")

    def start(self) -> None:
        """
        Start the pipeline by invoking start() on all stage actors,
        and launch the queue monitoring and scaling threads.
        """
        logger.debug("Starting pipeline: invoking start() on all stage actors...")
        start_refs = []
        for stage in self.stages:
            for actor in self.stage_actors.get(stage.name, []):
                if hasattr(actor, "start"):
                    logger.debug(f"Starting actor {actor} for stage {stage.name}")
                    start_refs.append(actor.start.remote())
        ray.get(start_refs)
        logger.debug("Pipeline started.")
        self._start_queue_monitoring()
        self._start_scaling()

    def stop(self) -> None:
        """
        Stop the pipeline by invoking stop() on all stage actors,
        and stop the queue monitoring and scaling threads.
        """
        logger.debug("Stopping pipeline: invoking stop() on all stage actors...")
        stop_refs = []
        for stage in self.stages:
            for actor in self.stage_actors.get(stage.name, []):
                if hasattr(actor, "stop"):
                    logger.debug(f"Stopping actor {actor} for stage {stage.name}")
                    stop_refs.append(actor.stop.remote())
        ray.get(stop_refs)
        logger.debug("Pipeline stopped.")
        self._stop_queue_monitoring()
        self._stop_scaling()


class UtilizationDisplay:
    """
    Helper class to display queue utilization snapshots in-place.
    """

    def __init__(self, refresh_rate: float = 2):
        """
        Initialize the UtilizationDisplay with a given refresh rate for updating the live display.

        Parameters
        ----------
        refresh_rate : float, optional
            The number of times per second to refresh the display, by default 2.
        """
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=refresh_rate, transient=False)
        self.live.start()

    def update(self, output_rows):
        """
        Update the live display with the latest queue utilization statistics.

        Parameters
        ----------
        output_rows : list of tuple
            A list of tuples containing utilization data for each stage.
        """
        table = Table(title="Queue Utilization Snapshot")
        table.add_column("Stage", justify="left", style="cyan", no_wrap=True)
        table.add_column("Replicas (current/max)", justify="left", style="magenta")
        table.add_column("Input Queue (occupancy/max)", justify="left", style="green")
        table.add_column("Scaling State", justify="left", style="yellow")
        table.add_column("Processing", justify="right", style="red")
        table.add_column("In Flight", justify="right", style="bright_blue")
        for row in output_rows:
            table.add_row(*row)
        self.live.update(table)

    def stop(self):
        """
        Stop and close the live display.
        """
        self.live.stop()
