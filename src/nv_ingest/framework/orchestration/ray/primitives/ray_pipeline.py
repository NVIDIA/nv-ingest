# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List
import ray
from pydantic import BaseModel
import logging

from nv_ingest.framework.orchestration.ray.edges.threaded_queue_edge import ThreadedQueueEdge

# Assume logger is already configured
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    """
    Information about a pipeline stage.

    Attributes
    ----------
    name : str
        Name of the stage.
    callable : Any
        A Ray remote actor class that implements the stage.
    config : BaseModel
        Configuration parameters for the stage.
    is_source : bool, optional
        Whether the stage is a source.
    is_sink : bool, optional
        Whether the stage is a sink.
    min_replicas : int, optional
        The minimum number of replicas for the stage.
        For sources and sinks this must be at least 1; for intermediate stages, it can be 0.
    max_replicas : int, optional
        The maximum number of replicas allowed for dynamic scaling.
    """

    name: str
    callable: Any  # Already a remote actor class
    config: BaseModel
    is_source: bool = False
    is_sink: bool = False
    min_replicas: int = 0
    max_replicas: int = 1


class RayPipeline:
    """
    A structured pipeline that supports source, intermediate, and sink stages.
    Stages are connected using ThreadedQueueEdge actors.
    """

    def __init__(self, scaling_threshold: float = 10.0, scaling_cooldown: int = 15) -> None:
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        # Store edge actor with its configured max size as a tuple (edge_actor, max_size)
        self.edge_queues: Dict[str, Any] = {}
        # Store queue utilization over time:
        # { edge_name: [ { "timestamp": float, "utilization": float }, ... ] }
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}
        # Scaling parameters and per-stage cooldown tracking.
        self.scaling_threshold: float = scaling_threshold
        self.scaling_cooldown: int = scaling_cooldown  # For scale down: number of cycles with zero utilization.
        self.under_threshold_cycles: Dict[str, int] = {}  # stage name -> count

        # New parameter: idle flag.
        self.idle: bool = True

        # Monitoring thread attributes.
        self._monitoring: bool = False
        self._monitor_thread: threading.Thread = None
        logger.debug("RayPipeline initialized with idle=True.")

    def add_source(
        self,
        *,
        name: str,
        source_actor: Any,
        config: BaseModel,
        min_replicas: int = 1,
        max_replicas: int = 1,
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
        self,
        *,
        name: str,
        stage_actor: Any,
        config: BaseModel,
        min_replicas: int = 0,
        max_replicas: int = 1,
    ) -> "RayPipeline":
        if min_replicas < 0:
            logger.warning(f"Stage '{name}': min_replicas cannot be negative. Overriding to 0.")
            min_replicas = 0
        stage_info = StageInfo(
            name=name,
            callable=stage_actor,
            config=config,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        self.stages.append(stage_info)
        logger.debug(f"Added intermediate stage: {stage_info}")
        return self

    def add_sink(
        self,
        *,
        name: str,
        sink_actor: Any,
        config: BaseModel,
        min_replicas: int = 1,
        max_replicas: int = 1,
    ) -> "RayPipeline":
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
        # Instantiate stage actors with initial replication count equal to min_replicas.
        for stage in self.stages:
            replicas = []
            for i in range(stage.min_replicas):
                actor_name = f"{stage.name}_{i}" if stage.min_replicas > 1 else stage.name
                logger.debug(f"Creating actor {actor_name} for stage {stage.name}")
                actor = stage.callable.options(name=actor_name, max_concurrency=100).remote(
                    config=stage.config, progress_engine_count=-1  # placeholder if needed
                )
                replicas.append(actor)
            # Even if min_replicas is 0, we store an empty list.
            self.stage_actors[stage.name] = replicas
            # Initialize scaling cooldown counter for this stage.
            self.under_threshold_cycles[stage.name] = 0
            logger.debug(f"Stage {stage.name} actors: {replicas}")

        logger.debug("Wiring up edges between stages...")
        wiring_refs = []  # Collect remote wiring calls.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                logger.debug(f"Creating edge actor {queue_name} with queue_size {queue_size}")
                edge_actor = ThreadedQueueEdge.options(name=queue_name, max_concurrency=100).remote(
                    max_size=queue_size, multi_reader=True, multi_writer=True
                )
                self.edge_queues[queue_name] = (edge_actor, queue_size)
                self.queue_stats[queue_name] = []

                # Wire output edges for the source stage, if any actors exist.
                for actor in self.stage_actors.get(from_stage, []):
                    logger.debug(f"Wiring output edge for actor {actor} in stage {from_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_output_edge.remote(edge_actor))
                # Wire input edges for the destination stage, if any actors exist.
                for actor in self.stage_actors.get(to_stage, []):
                    logger.debug(f"Wiring input edge for actor {actor} in stage {to_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_input_edge.remote(edge_actor))

        logger.debug("Waiting for all wiring calls to complete...")
        ray.get(wiring_refs)
        logger.debug("Pipeline build complete.")
        return self.stage_actors

    def _scale_stage(self, stage_name: str, new_replica_count: int) -> None:
        """
        Dynamically scale the given stage to new_replica_count by either adding new replicas
        or stopping existing ones. New actors are automatically wired to the appropriate edges
        and started after wiring completes.
        """
        current_replicas = self.stage_actors.get(stage_name, [])
        current_count = len(current_replicas)
        logger.debug(
            f"[Scale Stage] Stage '{stage_name}': current_count = {current_count},"
            f" target new_replica_count = {new_replica_count}"
        )
        stage_info = next((s for s in self.stages if s.name == stage_name), None)
        if stage_info is None:
            logger.error(f"[Scale Stage] Stage info for '{stage_name}' not found during scaling.")
            return

        if new_replica_count > current_count:
            logger.debug(
                f"[Scale Stage] Scaling UP stage '{stage_name}' from {current_count} to {new_replica_count} replicas."
            )
            for i in range(current_count, new_replica_count):
                actor_name = f"{stage_name}_{i}"
                logger.info(f"[Scale Stage] Creating new replica '{actor_name}' for stage '{stage_name}'")
                new_actor = stage_info.callable.options(name=actor_name, max_concurrency=10).remote(
                    config=stage_info.config, progress_engine_count=-1
                )
                wiring_refs = []
                # Wire new actor's output edges (where this stage is the source).
                if stage_name in self.connections:
                    for to_stage, _ in self.connections[stage_name]:
                        queue_name = f"{stage_name}_to_{to_stage}"
                        if queue_name in self.edge_queues:
                            edge_actor, _ = self.edge_queues[queue_name]
                            logger.debug(f"[Scale Stage] Wiring new actor '{actor_name}' output to edge '{queue_name}'")
                            wiring_refs.append(new_actor.set_output_edge.remote(edge_actor))
                        else:
                            logger.error(f"[Scale Stage] Output edge '{queue_name}' not found for stage '{stage_name}'")
                # Wire new actor's input edges (where this stage is the destination).
                for from_stage, conns in self.connections.items():
                    for to_stage, _ in conns:
                        if to_stage == stage_name:
                            queue_name = f"{from_stage}_to_{stage_name}"
                            if queue_name in self.edge_queues:
                                edge_actor, _ = self.edge_queues[queue_name]
                                logger.debug(
                                    f"[Scale Stage] Wiring new actor '{actor_name}' input to edge '{queue_name}'"
                                )
                                wiring_refs.append(new_actor.set_input_edge.remote(edge_actor))
                            else:
                                logger.error(
                                    f"[Scale Stage] Input edge '{queue_name}' not found for stage '{stage_name}'"
                                )
                if wiring_refs:
                    ray.get(wiring_refs)
                    logger.debug(f"[Scale Stage] Wiring complete for new actor '{actor_name}'")
                # Start the new actor.
                try:
                    if hasattr(new_actor, "start"):
                        logger.debug(
                            f"[Scale Stage] Invoking start() on new actor '{actor_name}' for stage '{stage_name}'"
                        )
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
            logger.info(
                f"[Scale Stage] Scaling UP complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )
        elif new_replica_count < current_count:
            logger.debug(
                f"[Scale Stage] Scaling DOWN stage '{stage_name}' from {current_count} to {new_replica_count} replicas."
            )
            remove_count = current_count - new_replica_count
            for _ in range(remove_count):
                actor = current_replicas.pop()  # Remove the last actor.
                logger.info(f"[Scale Stage] Stopping replica '{actor}' for stage '{stage_name}'")
                try:
                    ray.get(actor.stop.remote())
                    logger.debug(f"[Scale Stage] Successfully stopped replica '{actor}'")
                except Exception as e:
                    logger.error(f"[Scale Stage] Error stopping actor during scaling down: {e}")
            self.stage_actors[stage_name] = current_replicas
            logger.info(
                f"[Scale Stage] Scaling DOWN complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )

    def _perform_scaling(self) -> None:
        """
        Check the current utilization of input queues for each stage and decide
        whether to scale up or down. Scaling up is triggered if the maximum occupancy
        among the input edges exceeds the scaling_threshold. If idle is True at the first scale-up event,
        immediately scale all stages to half their max_replicas (but not below min_replicas) and set idle to False.
        Scaling down is triggered only after 15 consecutive cycles with zero utilization.
        """
        for stage in self.stages:
            stage_name = stage.name
            # Identify input edges for this stage.
            input_edges = [edge for edge in self.edge_queues.keys() if edge.endswith(f"_to_{stage_name}")]
            logger.debug(f"[Perform Scaling] Stage '{stage_name}': input_edges = {input_edges}")
            if not input_edges:
                logger.debug(f"[Perform Scaling] No input edges for stage '{stage_name}'; skipping scaling decision.")
                continue  # Possibly a source stage.
            max_util = 0.0
            for edge_name in input_edges:
                if self.queue_stats.get(edge_name):
                    last_util = self.queue_stats[edge_name][-1]["utilization"]
                    logger.debug(
                        f"[Perform Scaling] Stage '{stage_name}',"
                        f" edge '{edge_name}': last utilization = {last_util:.1f}%"
                    )
                    if last_util > max_util:
                        max_util = last_util
            logger.debug(
                f"[Perform Scaling] Stage '{stage_name}': maximum utilization among input edges = {max_util:.1f}%"
            )
            current_count = len(self.stage_actors.get(stage_name, []))
            logger.debug(
                f"[Perform Scaling] Stage '{stage_name}':"
                f" current replica count = {current_count}, scaling threshold = {self.scaling_threshold}%"
            )

            # Check for scale-up conditions.
            if max_util > self.scaling_threshold:
                # If idle is True and this is the first scale-up event, scale all stages immediately.
                if self.idle:
                    logger.info("[Perform Scaling] Idle flag set. Scaling all stages to half their max_replicas.")
                    for s in self.stages:
                        target = max(s.min_replicas, s.max_replicas // 2)
                        logger.info(
                            f"[Perform Scaling] Scaling stage '{s.name}' to {target} replicas due to idle trigger."
                        )
                        self._scale_stage(s.name, target)
                    self.idle = False
                    # Reset under_threshold_cycles for all stages.
                    for s in self.stages:
                        self.under_threshold_cycles[s.name] = 0
                else:
                    # Regular scale-up for the stage.
                    if current_count < stage.max_replicas:
                        new_count = min(current_count * 2 if current_count > 0 else 1, stage.max_replicas)
                        logger.info(
                            f"[Perform Scaling] Scaling UP stage '{stage_name}':"
                            f" occupancy {max_util:.1f}% exceeds threshold "
                            f"{self.scaling_threshold}%. Replicas {current_count} -> {new_count}"
                        )
                        self._scale_stage(stage_name, new_count)
                        self.under_threshold_cycles[stage_name] = 0
                    else:
                        logger.debug(
                            f"[Perform Scaling] Stage '{stage_name}' is already at maximum replicas ({current_count})."
                        )
            else:
                # Only scale down if the maximum utilization is exactly zero.
                if max_util == 0:
                    self.under_threshold_cycles[stage_name] = self.under_threshold_cycles.get(stage_name, 0) + 1
                    logger.debug(
                        f"[Perform Scaling] Stage '{stage_name}'"
                        f" under zero utilization for {self.under_threshold_cycles[stage_name]} cycles."
                    )
                    if self.under_threshold_cycles[stage_name] >= self.scaling_cooldown:
                        if current_count > stage.min_replicas:
                            new_count = max(current_count // 2, stage.min_replicas)
                            logger.info(
                                f"[Perform Scaling] Scaling DOWN stage '{stage_name}':"
                                f" zero occupancy for {self.scaling_cooldown} cycles."
                                f" Replicas {current_count} -> {new_count}"
                            )
                            self._scale_stage(stage_name, new_count)
                            self.under_threshold_cycles[stage_name] = 0
                        else:
                            logger.debug(
                                f"[Perform Scaling] Stage '{stage_name}'"
                                f" already at or below minimum replicas ({current_count})."
                            )
                else:
                    # If utilization is nonzero but below threshold, reset the counter.
                    self.under_threshold_cycles[stage_name] = 0

    def _monitor_queue_utilization(self, poll_interval: float = 10.0) -> None:
        """
        Runs in a separate thread to monitor the percent utilization (current_size vs max_size)
        of each edge queue. Issues all stats requests in parallel, processes them as a batch,
        updates self.queue_stats with timestamped values, outputs a formatted snapshot that includes:
        stage name, replica count (current/max), input queue occupancy, and current scaling state.
        Then, calls _perform_scaling() to adjust the stage replica counts.
        """
        logger.info("Queue monitoring thread started.")
        while self._monitoring:
            current_time = time.time()
            # Issue all get_stats remote calls concurrently.
            futures = {
                edge_name: edge_actor.get_stats.remote() for edge_name, (edge_actor, _) in self.edge_queues.items()
            }
            stats_results = ray.get(list(futures.values()))
            for edge_name, stats in zip(futures.keys(), stats_results):
                _, max_size = self.edge_queues[edge_name]
                current_size = stats.get("current_size", 0)
                utilization = (current_size / max_size) * 100 if max_size > 0 else 0
                logger.info(f"[Monitor] Edge '{edge_name}': {current_size}/{max_size} ({utilization:.1f}%) utilized.")
                self.queue_stats[edge_name].append({"timestamp": current_time, "utilization": utilization})

            # Generate periodic output for each stage.
            output_rows = []
            for stage in self.stages:
                stage_name = stage.name
                current_replica_count = len(self.stage_actors.get(stage_name, []))
                replicas_str = f"{current_replica_count}/{stage.max_replicas}"
                # Identify input edges for the stage.
                input_edges = [
                    edge_name for edge_name in self.edge_queues.keys() if edge_name.endswith(f"_to_{stage_name}")
                ]
                occupancy_list = []
                for edge_name in input_edges:
                    stats_list = self.queue_stats.get(edge_name, [])
                    if stats_list:
                        last_util = stats_list[-1]["utilization"]
                        _, max_size = self.edge_queues[edge_name]
                        occupancy = int((last_util / 100) * max_size)
                        occupancy_list.append(f"{occupancy}/{max_size}")
                    else:
                        occupancy_list.append("0/0")
                occupancy_str = ", ".join(occupancy_list) if occupancy_list else "N/A"
                scaling_state = "Pending"  # Placeholder for dynamic scaling state.
                output_rows.append((stage_name, replicas_str, occupancy_str, scaling_state))

            header = (
                f"{'Stage':<20} {'Replicas (current/max)':<25} "
                f"{'Input Queue (occupancy/max)':<30} {'Scaling State':<15}"
            )
            print("\nQueue Utilization Snapshot:")
            print(header)
            print("-" * len(header))
            for row in output_rows:
                print(f"{row[0]:<20} {row[1]:<25} {row[2]:<30} {row[3]:<15}")

            # Call the scaling decision function.
            self._perform_scaling()

            time.sleep(poll_interval)
        logger.info("Queue monitoring thread stopped.")

    def _start_queue_monitoring(self) -> None:
        """
        Starts the queue monitoring thread.
        """
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_queue_utilization, daemon=True)
            self._monitor_thread.start()
            logger.debug("Queue monitoring thread launched.")

    def _stop_queue_monitoring(self) -> None:
        """
        Stops the queue monitoring thread.
        """
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread is not None:
                self._monitor_thread.join()
            logger.debug("Queue monitoring thread stopped.")

    def start(self) -> None:
        """
        Start the pipeline by invoking start() on all stage actors concurrently,
        and launch the queue monitoring thread.
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

    def stop(self) -> None:
        """
        Stop the pipeline by invoking stop() on all stage actors concurrently,
        and stop the queue monitoring thread.
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
