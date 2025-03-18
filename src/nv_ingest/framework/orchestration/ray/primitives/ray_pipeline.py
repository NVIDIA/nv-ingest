# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import ray

logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    """
    Information about a pipeline stage.

    Parameters
    ----------
    name : str
        Name of the stage.
    callable : Any
        A callable (typically a Ray remote actor class) that implements the stage.
    config : Dict[str, Any]
        Configuration parameters for the stage.
    is_source : bool, optional
        Whether the stage is a source. Default is False.
    is_sink : bool, optional
        Whether the stage is a sink. Default is False.
    """

    name: str
    callable: Any  # Already a remote actor class
    config: Dict[str, Any]
    is_source: bool = False
    is_sink: bool = False


@ray.remote
class FixedSizeQueue:
    """
    A fixed-size queue actor that uses an asyncio.Queue internally.

    Parameters
    ----------
    max_size : int
        The maximum size of the queue.
    """

    def __init__(self, max_size: int) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.stats: Dict[str, int] = {"put_count": 0, "get_count": 0, "queue_full_count": 0}

    async def put(self, item: Any) -> bool:
        """
        Put an item into the queue.

        Parameters
        ----------
        item : Any
            The item to enqueue.

        Returns
        -------
        bool
            True if the item was enqueued successfully.
        """
        start_time = time.time()
        _ = start_time
        if self.queue.full():
            self.stats["queue_full_count"] += 1
        await self.queue.put(item)
        self.stats["put_count"] += 1
        return True

    async def get(self) -> Any:
        """
        Retrieve an item from the queue.

        Returns
        -------
        Any
            The next item in the queue.
        """
        item = await self.queue.get()
        self.stats["get_count"] += 1
        return item

    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics for the queue.

        Returns
        -------
        Dict[str, int]
            A dictionary with statistics (put_count, get_count, queue_full_count, current_size).
        """
        self.stats["current_size"] = self.queue.qsize()
        return self.stats


@ray.remote
class QueueConsumer:
    """
    A consumer actor that continuously pulls messages from a fixed-size queue and forwards
    them to a destination stage.
    """

    async def run(self, queue: Any, destination: Any) -> None:
        """
        Continuously retrieve messages from the queue and forward them to the destination.

        Parameters
        ----------
        queue : Any
            The fixed-size queue actor from which to retrieve messages.
        destination : Any
            The destination stage actor to which messages will be forwarded.
        """
        while True:
            try:
                # Await the queue's get() call asynchronously.
                control_message = await queue.get.remote()
                if control_message is None:
                    await asyncio.sleep(0.1)
                    continue
                await destination.process.remote(control_message)
            except Exception as e:
                logger.exception(f"Error in consumer: {e}")
                await asyncio.sleep(0.1)


class RayPipeline:
    """
    A simplified pipeline supporting a source, intermediate processing stages, and a sink.
    Stages are connected via fixed-size queues and dedicated consumer actors.

    Attributes
    ----------
    stages : List[StageInfo]
        List of stage definitions.
    connections : Dict[str, List[tuple]]
        Mapping from source stage name to list of (destination stage name, queue size) tuples.
    stage_actors : Dict[str, List[Any]]
        Mapping from stage name to list of instantiated Ray actor handles.
    edge_queues : Dict[str, Any]
        Mapping from edge (queue) name to fixed-size queue actor.
    consumers : Dict[str, List[Any]]
        Mapping from destination stage name to list of consumer actor handles.
    """

    def __init__(self) -> None:
        """
        Initialize a new RayPipeline instance.
        """
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # queue_name -> FixedSizeQueue actor
        self.consumers: Dict[str, List[Any]] = {}  # to_stage -> list of consumer actors

    def add_source(self, name: str, source_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add a source stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the source stage.
        source_actor : Any
            A Ray remote actor class representing the source stage.
        **config : dict
            Additional configuration parameters for the source stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=source_actor, config=config, is_source=True))
        return self

    def add_stage(self, name: str, stage_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add an intermediate processing stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the stage.
        stage_actor : Any
            A Ray remote actor class representing the processing stage.
        **config : dict
            Additional configuration parameters for the stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=stage_actor, config=config))
        return self

    def add_sink(self, name: str, sink_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add a sink stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the sink stage.
        sink_actor : Any
            A Ray remote actor class representing the sink stage.
        **config : dict
            Additional configuration parameters for the sink stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=sink_actor, config=config, is_sink=True))
        return self

    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        """
        Create an edge between two stages using a fixed-size queue.

        Parameters
        ----------
        from_stage : str
            Name of the source stage.
        to_stage : str
            Name of the destination stage.
        queue_size : int, optional
            The size of the fixed-size queue. Default is 100.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).

        Raises
        ------
        ValueError
            If either the from_stage or to_stage is not found in the pipeline.
        """
        if from_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        return self

    def build(self) -> Dict[str, List[Any]]:
        """
        Build the pipeline by instantiating actors for each stage and wiring up edges with fixed-size queues.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping stage names to lists of instantiated Ray actor handles.
        """
        # Create actor instances for each stage.
        for stage in self.stages:
            actor = stage.callable.options(name=stage.name).remote(**stage.config)
            self.stage_actors[stage.name] = [actor]

        # Create fixed-size queues and wire up edges.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                queue_actor = FixedSizeQueue.options(name=queue_name).remote(queue_size)
                self.edge_queues[queue_name] = queue_actor

                # For each upstream actor, set its downstream queue for this edge.
                for actor in self.stage_actors[from_stage]:
                    ray.get(actor.set_output_queue.remote(queue_actor))

                # Create a consumer actor for this edge.
                consumer = QueueConsumer.remote()
                self.consumers.setdefault(to_stage, []).append(consumer)
                consumer.run.remote(queue_actor, self.stage_actors[to_stage][0])
        return self.stage_actors

    def start(self) -> None:
        """
        Start the pipeline by invoking the start() method on all source actors, if available.

        Returns
        -------
        None
        """
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "start"):
                        ray.get(actor.start.remote())

    def stop(self) -> None:
        """
        Stop the pipeline by invoking the stop() method on all source actors, if available.

        Returns
        -------
        None
        """
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "stop"):
                        ray.get(actor.stop.remote())
