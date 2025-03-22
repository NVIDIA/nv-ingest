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

# Assume logger is already configured
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

    def __init__(self) -> None:
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        # Store edge actor with its configured max size as a tuple (edge_actor, max_size)
        self.edge_queues: Dict[str, Any] = {}
        # Store queue utilization over time: { edge_name: [ { "timestamp": float, "utilization": float }, ... ] }
        self.queue_stats: Dict[str, List[Dict[str, float]]] = {}
        # Monitoring thread attributes.
        self._monitoring: bool = False
        self._monitor_thread: threading.Thread = None
        logger.debug("RayPipeline initialized.")

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
                actor = stage.callable.options(name=actor_name, max_concurrency=10).remote(
                    config=stage.config, progress_engine_count=-1  # Placeholder
                )
                replicas.append(actor)
            self.stage_actors[stage.name] = replicas
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

                for actor in self.stage_actors[from_stage]:
                    logger.debug(f"Wiring output edge for actor {actor} in stage {from_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_output_edge.remote(edge_actor))
                for actor in self.stage_actors[to_stage]:
                    logger.debug(f"Wiring input edge for actor {actor} in stage {to_stage} to edge {queue_name}")
                    wiring_refs.append(actor.set_input_edge.remote(edge_actor))

        logger.debug("Waiting for all wiring calls to complete...")
        ray.get(wiring_refs)
        logger.debug("Pipeline build complete.")
        return self.stage_actors

    def _monitor_queue_utilization(self, poll_interval: float = 1.0) -> None:
        """
        Internal method that runs in a separate thread to monitor the percent utilization
        (current_size vs max_size) of each edge queue. Issues all stats requests in parallel,
        processes them as a batch, updates the self.queue_stats variable with timestamped utilization values,
        and outputs a formatted, execution-order sorted list including stage name, replica count (current/max),
        input queue occupancy vs max, and current scaling state.
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
                logger.info(f"Edge '{edge_name}': {current_size}/{max_size} ({utilization:.1f}%) utilized.")
                self.queue_stats[edge_name].append({"timestamp": current_time, "utilization": utilization})

            # Generate periodic output for each stage.
            output_rows = []
            for stage in self.stages:
                stage_name = stage.name
                current_replica_count = len(self.stage_actors.get(stage.name, []))
                # Show current replica count over max_replicas.
                replicas_str = f"{current_replica_count}/{stage.max_replicas}"
                # Identify all input edges for the stage.
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
                scaling_state = "Static"  # Placeholder for future dynamic scaling logic.
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
        Start the pipeline by invoking the start() method on all stage actors concurrently,
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
        Stop the pipeline by invoking the stop() method on all stage actors concurrently
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


# Context remains unchanged: ThreadedQueueEdge, etc.
@ray.remote
class ThreadedQueueEdge:
    """
    A threaded implementation of RayActorEdge using queue.Queue for thread safety.
    """

    def __init__(self, max_size: int, multi_reader: bool = False, multi_writer: bool = False) -> None:
        from queue import Queue
        from threading import Lock

        self.queue = Queue(maxsize=max_size)
        self.stats = {"write_count": 0, "read_count": 0, "queue_full_count": 0}
        self.stats_lock = Lock()
        logger.info(
            f"ThreadedQueueEdge initialized with max_size={max_size}, "
            f"multi_reader={multi_reader}, multi_writer={multi_writer}"
        )

    def write(self, item: Any) -> bool:
        if self.queue.full():
            with self.stats_lock:
                self.stats["queue_full_count"] += 1
            logger.info("Queue is full. Incrementing queue_full_count.")
        logger.info("Attempting to put item into the queue.")
        self.queue.put(item)
        with self.stats_lock:
            self.stats["write_count"] += 1
            logger.info(f"Item written to queue. New write_count: {self.stats['write_count']}")
        return True

    def read(self) -> Any:
        logger.info("Attempting to get item from the queue.")
        item = self.queue.get()
        with self.stats_lock:
            self.stats["read_count"] += 1
            logger.info(f"Item read from queue. New read_count: {self.stats['read_count']}")
        return item

    def get_stats(self) -> Dict[str, int]:
        with self.stats_lock:
            self.stats["current_size"] = self.queue.qsize()
            logger.info(f"Getting stats: {self.stats}")
            return self.stats.copy()
