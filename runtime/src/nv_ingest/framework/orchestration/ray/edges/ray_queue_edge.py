# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(Devin): Early prototype. Not currently used anywhere

import logging
from typing import Any, Dict
from threading import Lock

import ray
from ray.util.queue import Queue

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_edge_base import RayActorEdge

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@ray.remote
class RayQueueEdge(RayActorEdge):
    """
    A RayActorEdge implementation using ray.util.queue.Queue for improved efficiency.
    """

    def __init__(self, max_size: int, multi_reader: bool = False, multi_writer: bool = False) -> None:
        super().__init__(max_size, multi_reader, multi_writer)
        # Use Ray's distributed queue
        self.queue = Queue(maxsize=max_size)
        self.stats = {"write_count": 0, "read_count": 0, "queue_full_count": 0}
        # Dedicated lock for stats updates
        self.stats_lock = Lock()
        logger.info(
            f"ThreadedQueueEdge initialized with max_size={max_size}, "
            f"multi_reader={multi_reader}, multi_writer={multi_writer}"
        )

    # TODO(Devin): Think about adding timeouts to queue read/writes here. Stage loops already have timeouts, but
    # adding timeouts here would allow for more graceful handling of queue issues.
    def write(self, item: Any) -> bool:
        """
        Write an item into the queue synchronously.
        """
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
        """
        Read an item from the queue synchronously.
        """
        logger.info("Attempting to get item from the queue.")
        item = self.queue.get()
        with self.stats_lock:
            self.stats["read_count"] += 1
            logger.info(f"Item read from queue. New read_count: {self.stats['read_count']}")
        return item

    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics for the queue.
        """
        with self.stats_lock:
            self.stats["current_size"] = self.queue.qsize()
            logger.info(f"Getting stats: {self.stats}")
            return self.stats.copy()
