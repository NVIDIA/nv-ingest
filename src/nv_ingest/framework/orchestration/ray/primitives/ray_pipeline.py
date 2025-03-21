# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from pydantic import BaseModel
from nv_ingest.framework.orchestration.ray.edges.threaded_queue_edge import ThreadedQueueEdge
from typing import Any, Dict, List
import ray


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("RayPipelineHarness")


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
    progress_engine_count : int, optional
        The number of actor replicas to deploy for this stage.
    """

    name: str
    callable: Any  # Already a remote actor class
    config: BaseModel
    is_source: bool = False
    is_sink: bool = False
    progress_engine_count: int = 1  # default to 1 replica


class RayPipeline:
    """
    A structured pipeline that supports source, intermediate, and sink stages.
    Stages are connected using ThreadedQueueEdge actors, and stage implementations
    are expected to be thread–based (synchronous) rather than async.

    Attributes
    ----------
    stages : List[StageInfo]
        List of stage definitions.
    connections : Dict[str, List[tuple]]
        Mapping from stage name to a list of tuples (destination stage name, queue size).
    stage_actors : Dict[str, List[Any]]
        Mapping from stage name to lists of instantiated Ray actor handles.
    edge_queues : Dict[str, Any]
        Mapping from edge name to a ThreadedQueueEdge actor.
    """

    def __init__(self) -> None:
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # edge_name -> ThreadedQueueEdge actor
        logger.debug("RayPipeline initialized.")

    def add_source(
        self,
        *,
        name: str,
        source_actor: Any,
        config: BaseModel,
        progress_engine_count: int = 1,
    ) -> "RayPipeline":
        stage_info = StageInfo(
            name=name,
            callable=source_actor,
            config=config,
            is_source=True,
            progress_engine_count=progress_engine_count,
        )
        self.stages.append(stage_info)
        logger.debug(f"Added source stage: {stage_info}")
        return self

    def add_stage(
        self, *, name: str, stage_actor: Any, config: BaseModel, progress_engine_count: int = 1
    ) -> "RayPipeline":
        stage_info = StageInfo(
            name=name, callable=stage_actor, config=config, progress_engine_count=progress_engine_count
        )
        self.stages.append(stage_info)
        logger.debug(f"Added intermediate stage: {stage_info}")
        return self

    def add_sink(
        self, *, name: str, sink_actor: Any, config: BaseModel, progress_engine_count: int = 1
    ) -> "RayPipeline":
        stage_info = StageInfo(
            name=name,
            callable=sink_actor,
            config=config,
            is_sink=True,
            progress_engine_count=progress_engine_count,
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
        # Instantiate stage actors with replication.
        for stage in self.stages:
            replicas = []
            for i in range(stage.progress_engine_count):
                actor_name = f"{stage.name}_{i}" if stage.progress_engine_count > 1 else stage.name
                logger.debug(f"Creating actor {actor_name} for stage {stage.name}")
                # TODO(Devin) Max max concurrency a function of replication count.
                actor = stage.callable.options(name=actor_name, max_concurrency=10).remote(
                    config=stage.config, progress_engine_count=stage.progress_engine_count
                )
                replicas.append(actor)
            self.stage_actors[stage.name] = replicas
            logger.debug(f"Stage {stage.name} actors: {replicas}")

        logger.debug("Wiring up edges between stages...")
        wiring_refs = []  # List to collect all remote wiring calls.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                logger.debug(f"Creating edge actor {queue_name} with queue_size {queue_size}")
                # TODO(Devin) Max max concurrency a function of replication count.
                edge_actor = ThreadedQueueEdge.options(name=queue_name, max_concurrency=100).remote(
                    max_size=queue_size, multi_reader=True, multi_writer=True
                )
                self.edge_queues[queue_name] = edge_actor

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

    def start(self) -> None:
        """
        Start the pipeline by invoking the start() method on all stage actors concurrently.
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

    def stop(self) -> None:
        """
        Stop the pipeline by invoking the stop() method on all stage actors concurrently.
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
