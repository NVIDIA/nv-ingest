# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Dict, Any

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_edge_base import RayActorEdge


@ray.remote
class AsyncQueueEdge(RayActorEdge):
    """
    A Ray actor edge that uses an asyncio.Queue internally.
    This edge supports configurable maximum size, as well as options for multiple readers and multiple writers.
    """

    def __init__(self, max_size: int, multi_reader: bool = False, multi_writer: bool = False) -> None:
        super().__init__(max_size, multi_reader, multi_writer)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.stats: Dict[str, int] = {"write_count": 0, "read_count": 0, "queue_full_count": 0}

    async def write(self, item: Any) -> bool:
        """
        Write an item into the queue.

        If the queue is full, the 'queue_full_count' statistic is incremented.

        Returns
        -------
        bool
            True if the item was enqueued successfully.
        """
        if self.queue.full():
            self.stats["queue_full_count"] += 1
        await self.queue.put(item)
        self.stats["write_count"] += 1
        return True

    async def read(self) -> Any:
        """
        Read an item from the queue.

        Returns
        -------
        Any
            The next item in the queue.
        """
        item = await self.queue.get()
        self.stats["read_count"] += 1
        return item

    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics for the queue.

        Returns
        -------
        Dict[str, int]
            A dictionary with statistics, including the current queue size.
        """
        self.stats["current_size"] = self.queue.qsize()
        return self.stats
