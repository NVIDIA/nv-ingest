import asyncio
import ray
import logging
from typing import Any, Dict

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_edge_base import RayActorEdge

logger = logging.getLogger(__name__)

# TODO(Devin): Early prototype. Not currently used anywhere


@ray.remote
class AsyncQueueEdge(RayActorEdge):
    """
    An asynchronous implementation of RayActorEdge using asyncio.Queue for thread safety.
    """

    def __init__(self, max_size: int, multi_reader: bool = False, multi_writer: bool = False) -> None:
        super().__init__(max_size, multi_reader, multi_writer)
        self.queue = asyncio.Queue(maxsize=max_size)
        self.stats = {"write_count": 0, "read_count": 0, "queue_full_count": 0}
        # Use a dedicated asyncio lock for updating stats.
        self.stats_lock = asyncio.Lock()
        logger.info(
            f"AsyncQueueEdge initialized with max_size={max_size},"
            f" multi_reader={multi_reader}, multi_writer={multi_writer}"
        )

    async def write(self, item: Any) -> bool:
        """
        Write an item into the edge asynchronously.
        """
        if self.queue.full():
            async with self.stats_lock:
                self.stats["queue_full_count"] += 1
            logger.info("Queue is full. Incrementing queue_full_count.")
        logger.info("Attempting to put item into the queue.")
        await self.queue.put(item)
        async with self.stats_lock:
            self.stats["write_count"] += 1
            logger.info(f"Item written to queue. New write_count: {self.stats['write_count']}")
        return True

    async def read(self) -> Any:
        """
        Read an item from the edge asynchronously.
        """
        logger.info("Attempting to get item from the queue.")
        item = await self.queue.get()
        async with self.stats_lock:
            self.stats["read_count"] += 1
            logger.info(f"Item read from queue. New read_count: {self.stats['read_count']}")
        return item

    async def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics for the queue.
        """
        async with self.stats_lock:
            self.stats["current_size"] = self.queue.qsize()
            logger.info(f"Getting stats: {self.stats}")
            return self.stats.copy()
