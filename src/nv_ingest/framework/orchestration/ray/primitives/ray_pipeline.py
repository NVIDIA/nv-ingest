# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
import uuid  # For generating unique actor names
from dataclasses import dataclass
from typing import Any, Dict, List
import ray
from pydantic import BaseModel
import logging

from rich.console import Console
from rich.table import Table
from rich.live import Live

from nv_ingest.framework.orchestration.ray.edges.threaded_queue_edge import ThreadedQueueEdge

# Assume logger is already configured
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
    Stages are connected using ThreadedQueueEdge actors.
    """

    def __init__(self, scaling_threshold: float = 10.0, scaling_cooldown: int = 10) -> None:
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # edge name -> (edge_actor, max_size)
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}
        self.scaling_threshold: float = scaling_threshold
        self.scaling_cooldown: int = scaling_cooldown
        self.under_threshold_cycles: Dict[str, int] = {}  # stage name -> cycle count

        self.idle: bool = True
        self.scaling_state: Dict[str, str] = {}

        # Thread attributes for monitoring and scaling.
        self._monitoring: bool = False
        self._monitor_thread: threading.Thread = None

        # New attributes for a dedicated scaling thread.
        self._scaling_monitoring: bool = False
        self._scaling_thread: threading.Thread = None

        logger.debug("RayPipeline initialized with idle=True.")

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

        logger.debug("Wiring up edges between stages...")
        wiring_refs = []
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                logger.debug(f"Creating edge actor {queue_name} with queue_size {queue_size}")
                edge_actor = ThreadedQueueEdge.options(name=queue_name, max_concurrency=100).remote(
                    max_size=queue_size, multi_reader=True, multi_writer=True
                )
                self.edge_queues[queue_name] = (edge_actor, queue_size)
                self.queue_stats[queue_name] = []

                for actor in self.stage_actors.get(from_stage, []):
                    logger.debug(f"Wiring output edge for actor {actor} in stage {from_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_output_edge.remote(edge_actor))
                for actor in self.stage_actors.get(to_stage, []):
                    logger.debug(f"Wiring input edge for actor {actor} in stage {to_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_input_edge.remote(edge_actor))
        logger.debug("Waiting for all wiring calls to complete...")
        ray.get(wiring_refs)
        logger.debug("Pipeline build complete.")
        return self.stage_actors

    def _scale_stage(self, stage_name: str, new_replica_count: int) -> None:
        """
        Dynamically scale the given stage to new_replica_count.
        For scaling up, ensure that the total does not exceed max_replicas.
        Allow scaling down regardless of the current count.
        """
        current_replicas = self.stage_actors.get(stage_name, [])
        current_count = len(current_replicas)
        stage_info = next((s for s in self.stages if s.name == stage_name), None)
        if stage_info is None:
            logger.error(f"[Scale Stage] Stage info for '{stage_name}' not found during scaling.")
            return

        if new_replica_count > current_count:
            # Scaling UP: cap at max_replicas.
            if current_count >= stage_info.max_replicas:
                logger.debug(
                    f"[Scale Stage] Stage '{stage_name}' already at max replicas ({current_count}). Skipping scale-up."
                )
                return
            target_count = min(new_replica_count, stage_info.max_replicas)
            logger.debug(
                f"[Scale Stage] Scaling UP stage '{stage_name}' from {current_count} to {target_count} replicas."
            )
            for _ in range(target_count - current_count):
                actor_name = f"{stage_name}_{uuid.uuid4()}"
                logger.info(f"[Scale Stage] Creating new replica '{actor_name}' for stage '{stage_name}'")
                new_actor = stage_info.callable.options(name=actor_name, max_concurrency=10).remote(
                    config=stage_info.config, progress_engine_count=-1
                )
                wiring_refs = []
                if stage_name in self.connections:
                    for to_stage, _ in self.connections[stage_name]:
                        queue_name = f"{stage_name}_to_{to_stage}"
                        if queue_name in self.edge_queues:
                            edge_actor, _ = self.edge_queues[queue_name]
                            logger.debug(f"[Scale Stage] Wiring new actor '{actor_name}' output to edge '{queue_name}'")
                            wiring_refs.append(new_actor.set_output_edge.remote(edge_actor))
                        else:
                            logger.error(f"[Scale Stage] Output edge '{queue_name}' not found for stage '{stage_name}'")
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
            logger.info(
                f"[Scale Stage] Scaling UP complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )
        elif new_replica_count < current_count:
            # Scaling DOWN: always allow scaling down.
            logger.debug(
                f"[Scale Stage] Scaling DOWN stage '{stage_name}' from {current_count} to {new_replica_count} replicas."
            )
            remove_count = current_count - new_replica_count
            stopped_actors = []
            stop_futures = []
            for _ in range(remove_count):
                actor = current_replicas.pop()
                stopped_actors.append(actor)
                logger.info(f"[Scale Stage] Stopping replica '{actor}' for stage '{stage_name}'")
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
            logger.info(
                f"[Scale Stage] Scaling DOWN complete for stage '{stage_name}'."
                f" New replica count: {len(current_replicas)}"
            )

    def _idle_scale_all_stages(self) -> None:
        """
        For idle-triggered scale-up, create missing replicas for every stage, capped at max_replicas.
        For each stage, target replica count = min(max(min_replicas, max_replicas // 2), max_replicas).
        """
        logger.info("[Idle Scale] Starting idle scale-up for all stages.")
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
                    logger.info(f"[Idle Scale] Creating new replica '{actor_name}' for stage '{stage_name}'")
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
                            edge_actor, _ = self.edge_queues[queue_name]
                            logger.debug(f"[Idle Scale] Wiring new actor '{new_actor}' output to edge '{queue_name}'")
                            wiring_refs.append(new_actor.set_output_edge.remote(edge_actor))
                        else:
                            logger.error(f"[Idle Scale] Output edge '{queue_name}' not found for stage '{stage_name}'")
                for from_stage, conns in self.connections.items():
                    for to_stage, _ in conns:
                        if to_stage == stage_name:
                            queue_name = f"{from_stage}_to_{stage_name}"
                            if queue_name in self.edge_queues:
                                edge_actor, _ = self.edge_queues[queue_name]
                                logger.debug(
                                    f"[Idle Scale] Wiring new actor '{new_actor}' input to edge '{queue_name}'"
                                )
                                wiring_refs.append(new_actor.set_input_edge.remote(edge_actor))
                            else:
                                logger.error(
                                    f"[Idle Scale] Input edge '{queue_name}' not found for stage '{stage_name}'"
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
        logger.info("[Idle Scale] Idle scale-up complete; all stages scaled to target replica counts.")

    def _perform_scaling(self) -> None:
        """
        Check the current utilization of input queues for each stage and decide whether
        to scale up or down. If a stage's input exceeds the threshold, we scale that stage
        (or trigger idle scale-up if idle). However, if any stage is scaled up on this pass,
        then downstream stages are allowed to scale up but are prevented from scaling down,
        in order to avoid 'sloshing'.
        """
        upscale_triggered = False
        for stage in self.stages:
            stage_name = stage.name
            input_edges = [edge for edge in self.edge_queues.keys() if edge.endswith(f"_to_{stage_name}")]
            logger.debug(f"[Perform Scaling] Stage '{stage_name}': input_edges = {input_edges}")
            if not input_edges:
                logger.debug(f"[Perform Scaling] No input edges for stage '{stage_name}'; skipping scaling decision.")
                continue

            max_util = 0.0
            for edge_name in input_edges:
                if self.queue_stats.get(edge_name):
                    last_util = self.queue_stats[edge_name][-1]["utilization"]
                    logger.debug(
                        f"[Perform Scaling] Stage '{stage_name}', edge '{edge_name}':"
                        f" last utilization = {last_util:.1f}%"
                    )
                    if last_util > max_util:
                        max_util = last_util
            logger.debug(
                f"[Perform Scaling] Stage '{stage_name}': maximum utilization among input edges = {max_util:.1f}%"
            )
            current_count = len(self.stage_actors.get(stage_name, []))
            logger.debug(
                f"[Perform Scaling] Stage '{stage_name}': current replica count = {current_count},"
                f" threshold = {self.scaling_threshold}%"
            )

            if max_util > self.scaling_threshold:
                if self.idle:
                    logger.info("[Perform Scaling] Idle flag set. Triggering bulk idle scale-up.")
                    self._idle_scale_all_stages()
                    self.idle = False
                else:
                    if current_count < stage.max_replicas:
                        new_count = min(current_count * 2 if current_count > 0 else 1, stage.max_replicas)
                        logger.info(
                            f"[Perform Scaling] Scaling UP stage '{stage_name}':"
                            f" occupancy {max_util:.1f}% exceeds threshold "
                            f"{self.scaling_threshold}%. Replicas {current_count} -> {new_count}"
                        )
                        self._scale_stage(stage_name, new_count)
                        upscale_triggered = True
                    else:
                        logger.debug(
                            f"[Perform Scaling] Stage '{stage_name}' is already at maximum replicas ({current_count})."
                        )
                self.under_threshold_cycles[stage_name] = 0
                # Do not block downstream stages entirely; allow them to scale up.
            else:
                if max_util == 0:
                    self.under_threshold_cycles[stage_name] = self.under_threshold_cycles.get(stage_name, 0) + 1
                    logger.debug(
                        f"[Perform Scaling] Stage '{stage_name}'"
                        f" under zero utilization for {self.under_threshold_cycles[stage_name]} cycles."
                    )
                    if self.under_threshold_cycles[stage_name] >= self.scaling_cooldown:
                        if current_count > stage.min_replicas:
                            new_count = max(current_count // 2, stage.min_replicas)
                            # Only perform a scale-down if no upstream stage was scaled up this pass.
                            if not upscale_triggered:
                                logger.info(
                                    f"[Perform Scaling] Scaling DOWN stage '{stage_name}':"
                                    f" zero occupancy for {self.scaling_cooldown} cycles."
                                    f" Replicas {current_count} -> {new_count}"
                                )
                                self._scale_stage(stage_name, new_count)
                            else:
                                logger.info(
                                    "[Perform Scaling] Skipping scale down for stage"
                                    f" '{stage_name}' due to upstream upscale."
                                )
                        else:
                            logger.debug(
                                f"[Perform Scaling] Stage '{stage_name}'"
                                f" already at or below minimum replicas ({current_count})."
                            )
                        self.under_threshold_cycles[stage_name] = 0
                        self.scaling_state[stage_name] = "Scaling Down"
                else:
                    self.under_threshold_cycles[stage_name] = 0
                    self.scaling_state[stage_name] = "Stable"

    def _monitor_queue_utilization(self, poll_interval: float = 10.0) -> None:
        """
        Monitor the percent utilization of each edge queue and update display.
        """
        display = UtilizationDisplay(refresh_rate=2)
        logger.info("Queue monitoring thread started.")
        while self._monitoring:
            current_time = time.time()
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

            output_rows = []
            for stage in self.stages:
                stage_name = stage.name
                current_replica_count = len(self.stage_actors.get(stage_name, []))
                replicas_str = f"{current_replica_count}/{stage.max_replicas}"
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
                scaling_state = self.scaling_state.get(stage_name, "Pending")
                output_rows.append((stage_name, replicas_str, occupancy_str, scaling_state))

            display.update(output_rows)
            time.sleep(poll_interval)
        display.stop()
        logger.info("Queue monitoring thread stopped.")

    def _start_queue_monitoring(self) -> None:
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_queue_utilization, daemon=True)
            self._monitor_thread.start()
            logger.debug("Queue monitoring thread launched.")

    def _stop_queue_monitoring(self) -> None:
        if self._monitoring:
            self._monitoring = False
            if self._monitor_thread is not None:
                self._monitor_thread.join()
            logger.debug("Queue monitoring thread stopped.")

    def _scaling_loop(self, poll_interval: float = 10.0) -> None:
        """
        Dedicated scaling loop that continuously calls _perform_scaling.
        """
        logger.info("Scaling thread started.")
        while self._scaling_monitoring:
            self._perform_scaling()
            time.sleep(poll_interval)
        logger.info("Scaling thread stopped.")

    def _start_scaling(self) -> None:
        if not self._scaling_monitoring:
            self._scaling_monitoring = True
            self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self._scaling_thread.start()
            logger.debug("Scaling thread launched.")

    def _stop_scaling(self) -> None:
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
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=refresh_rate, transient=False)
        self.live.start()

    def update(self, output_rows):
        table = Table(title="Queue Utilization Snapshot")
        table.add_column("Stage", justify="left", style="cyan", no_wrap=True)
        table.add_column("Replicas (current/max)", justify="left", style="magenta")
        table.add_column("Input Queue (occupancy/max)", justify="left", style="green")
        table.add_column("Scaling State", justify="left", style="yellow")
        for row in output_rows:
            table.add_row(row[0], row[1], row[2], row[3])
        self.live.update(table)

    def stop(self):
        self.live.stop()
