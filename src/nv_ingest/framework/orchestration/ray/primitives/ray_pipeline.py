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
        pid_ki: float = 0.0075,
        pid_kd: float = 0.05,
    ) -> None:
        """
        Initialize the RayPipeline instance.
        [Docstring unchanged...]
        """
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}
        self.scaling_threshold: float = scaling_threshold
        self.scaling_cooldown: int = scaling_cooldown
        self.under_threshold_cycles: Dict[str, int] = {}

        self.dynamic_memory_scaling = dynamic_memory_scaling
        self.dynamic_memory_threshold = dynamic_memory_threshold
        self.stage_memory_overhead: Dict[str, float] = {}

        self.idle: bool = True
        self.scaling_state: Dict[str, str] = {}

        self._monitoring: bool = False
        self._monitor_thread: threading.Thread = None
        self._scaling_monitoring: bool = False
        self._scaling_thread: threading.Thread = None

        self._last_queue_flush = time.time()

        # New: store per-stage processing and in-flight statistics.
        self.stage_stats: Dict[str, Dict[str, int]] = {}

        total_system_memory = psutil.virtual_memory().total
        memory_threshold_mb = int(dynamic_memory_threshold * total_system_memory / (1024 * 1024))
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            max_replicas=100,
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
                    stage.max_replicas = max(max(1, stage.min_replicas), int(stage.max_replicas * ratio))
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
            # [Scaling-up branch remains unchanged...]
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
            # For each extra replica, simply issue a stop request and remove it from our actor list.
            for _ in range(remove_count):
                actor = current_replicas.pop()
                logger.debug(f"[Scale Stage] Sending stop request to replica '{actor}' for stage '{stage_name}'")
                actor.stop.remote()
                # No waiting or manual kill; the actor will self-terminate when done.
            self.stage_actors[stage_name] = current_replicas
            self.scaling_state[stage_name] = "Scaling Down"
            logger.debug(
                f"[Scale Stage] Scaling DOWN complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )

    def _perform_scaling(self) -> None:
        """
        Use the PID controller to evaluate current metrics and adjust replica counts.

        Before evaluating PID scaling, if there is no in-flight traffic (global_in_flight == 0)
        and if at least 10 minutes have elapsed since the last queue flush, pause all source stages,
        swap in new queues for all pipeline stages (thereby resetting any accumulated overhead in
        the queue actors), and then resume the source stages.
        """

        # Compute global pipeline in-flight: sum of in_flight across all stages.
        global_in_flight = sum(self.stage_stats.get(s.name, {}).get("in_flight", 0) for s in self.stages)
        logger.debug(f"Global in-flight count: {global_in_flight}")

        # TODO(Devin): Implement dynamic queue cycling for memory management.
        # current_time = time.time()

        # Only perform a queue flush if there is no in-flight work and at least 10 minutes have passed.
        # if global_in_flight == 0 and (current_time - self._last_queue_flush >= 600) and False:
        #     logger.info("No in-flight tasks detected globally; initiating queue swap procedure.")

        #     # Pause all source stage actors.
        #     for stage in self.stages:
        #         if getattr(stage, "is_source", False):  # Assuming StageInfo has an 'is_source' flag.
        #             for actor in self.stage_actors.get(stage.name, []):
        #                 logger.info(f"Pausing source stage actor: {actor}")
        #                 ray.get(actor.pause.remote())

        #     # Create new queues and wire them in.
        #     wiring_refs = []
        #     for from_stage, conns in self.connections.items():
        #         for to_stage, queue_size in conns:
        #             queue_name = f"{from_stage}_to_{to_stage}"
        #             logger.debug(f"Creating new queue '{queue_name}' with size {queue_size}")
        #             new_queue = Queue(maxsize=queue_size)
        #             self.edge_queues[queue_name] = (new_queue, queue_size)
        #             self.queue_stats[queue_name] = []
        #             # Re-wire output for the from_stage.
        #             for actor in self.stage_actors.get(from_stage, []):
        #                 logger.debug(f"Setting new output queue for actor {actor} in stage {from_stage}")
        #                 wiring_refs.append(actor.set_output_queue.remote(new_queue))
        #             # Re-wire input for the to_stage.
        #             for actor in self.stage_actors.get(to_stage, []):
        #                 logger.debug(f"Setting new input queue for actor {actor} in stage {to_stage}")
        #                 wiring_refs.append(actor.set_input_queue.remote(new_queue))
        #     ray.get(wiring_refs)
        #     logger.info("Queue swap complete: new queues have been wired in.")

        #     # Update the last flush time.
        #     self._last_queue_flush = current_time

        #     # Resume all source stage actors.
        #     for stage in self.stages:
        #         if getattr(stage, "is_source", False):
        #             for actor in self.stage_actors.get(stage.name, []):
        #                 logger.info(f"Resuming source stage actor: {actor}")
        #                 ray.get(actor.resume.remote())
        #     logger.info("Source stages resumed after queue swap.")
        # else:
        #     if global_in_flight != 0:
        #         logger.debug("In-flight tasks detected; skipping queue flush.")
        #     else:
        #         logger.debug("Queue flush skipped; last flush occurred less than 10 minutes ago.")

        # Now, proceed with normal scaling calculations.
        stage_metrics: Dict[str, Dict[str, Any]] = {}
        for stage in self.stages:
            stage_name = stage.name
            # Compute queue depth from all input edges.
            input_edges = [ename for ename in self.edge_queues if ename.endswith(f"_to_{stage_name}")]
            queue_depth = sum(self.edge_queues[ename][0].qsize() for ename in input_edges) if input_edges else 0
            throughput = 0  # (Not measured)
            replicas = len(self.stage_actors.get(stage_name, []))
            stage_metrics[stage_name] = {
                "max_replicas": stage.max_replicas,
                "memory_usage": 5000,  # Hard-coded estimate.
                "min_replicas": stage.min_replicas,
                "queue_depth": queue_depth,
                "replicas": replicas,
                "target_queue_depth": 0,  # TODO: Parameterize.
                "throughput": throughput,
                "processing": self.stage_stats.get(stage_name, {}).get("processing", 0),
                "in_flight": self.stage_stats.get(stage_name, {}).get("in_flight", 0),
                "pipeline_in_flight": global_in_flight,
            }
        current_memory_usage_bytes = psutil.virtual_memory().used
        current_memory_usage_mb = current_memory_usage_bytes / (1024 * 1024)
        adjustments = self.pid_controller.update(stage_metrics, current_memory_usage_mb)

        # Launch scaling operations concurrently.
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
            concurrent.futures.wait(scaling_futures)

    def _monitor_queue_utilization(self, poll_interval: float = 10.0) -> None:
        """
        Monitor distributed queue utilization and store per-stage processing and in-flight stats.
        """
        display = UtilizationDisplay(refresh_rate=2)
        logger.debug("Queue monitoring thread started.")
        while self._monitoring:
            current_time = time.time()
            # Update queue stats.
            for edge_name, (edge_queue, max_size) in self.edge_queues.items():
                current_size = edge_queue.qsize()
                utilization = (current_size / max_size) * 100 if max_size > 0 else 0
                logger.debug(f"[Monitor] Queue '{edge_name}': {current_size}/{max_size} ({utilization:.1f}%) utilized.")
                self.queue_stats[edge_name].append({"timestamp": current_time, "utilization": utilization})
            # Compute per-stage stats.
            output_rows = []  # Build output for display.
            for stage in self.stages:
                stage_name = stage.name
                current_replicas = self.stage_actors.get(stage_name, [])
                processing_count = 0
                for actor in current_replicas:
                    try:
                        stats = ray.get(actor.get_stats.remote())
                        processing_count += int(stats.get("active_processing", 0))
                    except Exception as e:
                        logger.error(f"Error fetching stats for actor {actor} in stage {stage_name}: {e}")
                input_edges = [ename for ename in self.edge_queues if ename.endswith(f"_to_{stage_name}")]
                total_queued = sum(self.edge_queues[ename][0].qsize() for ename in input_edges) if input_edges else 0
                stage_in_flight = processing_count + total_queued
                # Save stats for scaling.
                self.stage_stats[stage_name] = {"processing": processing_count, "in_flight": stage_in_flight}
                # Construct a display row.
                replicas_str = f"{len(current_replicas)}/{stage.max_replicas}"
                occupancy_str = (
                    ", ".join(
                        f"{self.edge_queues[ename][0].qsize()}/{self.edge_queues[ename][1]}" for ename in input_edges
                    )
                    if input_edges
                    else "N/A"
                )
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
            # Optionally, add a global row.
            global_processing = sum(s.get("processing", 0) for s in self.stage_stats.values())
            global_in_flight = sum(s.get("in_flight", 0) for s in self.stage_stats.values())
            output_rows.append(
                ("[bold]Total Pipeline[/bold]", "", "", "", str(global_processing), str(global_in_flight))
            )
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
