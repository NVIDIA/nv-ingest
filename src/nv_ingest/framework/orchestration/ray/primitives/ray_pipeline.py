# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio

import ray
from typing import Any, Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    name: str
    callable: Any  # Already a remote actor class
    config: Dict[str, Any]
    is_source: bool = False
    is_sink: bool = False


class RayPipeline:
    """
    A simplified pipeline supporting a source, intermediate processing stages, and a sink.
    Stages are connected via fixed-size queues and dedicated consumer actors.
    """

    def __init__(self):
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # queue_name -> FixedSizeQueue actor
        self.consumers: Dict[str, List[Any]] = {}  # to_stage -> list of consumer actors

    def add_source(self, name: str, source_actor: Any, **config) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=source_actor, config=config, is_source=True))
        return self

    def add_stage(self, name: str, stage_actor: Any, **config) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=stage_actor, config=config))
        return self

    def add_sink(self, name: str, sink_actor: Any, **config) -> "RayPipeline":
        self.stages.append(StageInfo(name=name, callable=sink_actor, config=config, is_sink=True))
        return self

    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        if from_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        return self

    def build(self) -> Dict[str, List[Any]]:
        # Create actor instances for each stage.
        for stage in self.stages:
            actor = stage.callable.options(name=stage.name).remote(**stage.config)
            self.stage_actors[stage.name] = [actor]

        # Create fixed-size queues and wire up edges.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"

                @ray.remote
                class FixedSizeQueue:
                    def __init__(self, max_size):
                        self.queue = asyncio.Queue(maxsize=max_size)
                        self.stats = {"put_count": 0, "get_count": 0, "queue_full_count": 0}

                    async def put(self, item):
                        import time

                        start_time = time.time()
                        _ = start_time
                        if self.queue.full():
                            self.stats["queue_full_count"] += 1
                        await self.queue.put(item)
                        self.stats["put_count"] += 1
                        return True

                    async def get(self):
                        item = await self.queue.get()
                        self.stats["get_count"] += 1
                        return item

                    def get_stats(self):
                        self.stats["current_size"] = self.queue.qsize()
                        return self.stats

                queue_actor = FixedSizeQueue.options(name=queue_name).remote(queue_size)
                self.edge_queues[queue_name] = queue_actor

                # For each upstream actor, set its downstream queue for THIS edge.
                for actor in self.stage_actors[from_stage]:
                    # Instead of overwriting a shared attribute, you could have a mapping of edges if needed.
                    ray.get(actor.set_output_queue.remote(queue_actor))

                # Create a consumer actor for this edge.
                @ray.remote
                class QueueConsumer:
                    async def run(self, queue, destination):
                        while True:
                            try:
                                control_message = ray.get(queue.get.remote())
                                if control_message is None:
                                    await asyncio.sleep(0.1)
                                    continue
                                await destination.process.remote(control_message)
                            except Exception as e:
                                logger.exception(f"Error in consumer: {e}")
                                await asyncio.sleep(0.1)

                consumer = QueueConsumer.remote()
                self.consumers.setdefault(to_stage, []).append(consumer)
                consumer.run.remote(queue_actor, self.stage_actors[to_stage][0])
        return self.stage_actors

    def start(self) -> None:
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "start"):
                        ray.get(actor.start.remote())

    def stop(self) -> None:
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "stop"):
                        ray.get(actor.stop.remote())
