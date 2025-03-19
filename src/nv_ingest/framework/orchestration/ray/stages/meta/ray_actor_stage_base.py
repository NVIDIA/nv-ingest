# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import threading
import time
from abc import ABC, abstractmethod

import ray
from pydantic import BaseModel
from typing import Any
import logging

logger = logging.getLogger(__name__)


###############################################################################
# Abstract base class for Ray actor stages


class RayActorStage(ABC):
    """
    Abstract base class for a processing stage in a RayPipeline.

    This class defines a default get_input() method that returns a message from the input edge.
    Its processing loop (_processing_loop) always calls get_input() so that subclasses may override
    this behavior (for example, source stages can override get_input() to fetch messages externally).
    """

    def __init__(self, config: BaseModel, progress_engine_count: int) -> None:
        self.config = config
        self.progress_engine_count = progress_engine_count
        self.input_edge = None  # For non-source stages.
        self.output_edge = None
        self.running = False
        self.stats = {"processed": 0}
        self.start_time = None

    async def get_input(self) -> Any:
        """
        Default implementation for non-source stages:
        Reads a message from the input edge.
        """
        if self.input_edge is None:
            raise ValueError("Input edge not set")
        return await self.input_edge.read.remote()

    @abstractmethod
    async def on_data(self, control_message: Any) -> Any:
        """
        Process an incoming control message and return an updated control message.
        """
        pass

    async def _processing_loop(self) -> None:
        """
        Continuously fetches input via get_input(), processes it with on_data,
        and writes the updated message to the output edge.
        """
        while self.running:
            try:
                control_message = await self.get_input()
                if control_message is None:
                    await asyncio.sleep(self.config.poll_interval)
                    continue

                updated_cm = await self.on_data(control_message)
                if updated_cm and self.output_edge:
                    await self.output_edge.write.remote(updated_cm)
                self.stats["processed"] += 1
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                await asyncio.sleep(self.config.poll_interval)

    @ray.method(num_returns=1)
    def start(self) -> bool:
        """
        Start the stage by launching the processing loop in a background thread.
        """
        if self.running:
            return False
        self.running = True
        self.start_time = time.time()
        threading.Thread(target=lambda: asyncio.run(self._processing_loop()), daemon=True).start()
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        """
        Stop the stage.
        """
        self.running = False
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        """
        Return stage statistics.
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {"processed": self.stats["processed"], "elapsed": elapsed}

    @ray.method(num_returns=1)
    def set_input_edge(self, edge_handle: Any) -> bool:
        """
        Set the input edge for the stage.
        """
        self.input_edge = edge_handle
        return True

    @ray.method(num_returns=1)
    def set_output_edge(self, edge_handle: Any) -> bool:
        """
        Set the output edge for the stage.
        """
        self.output_edge = edge_handle
        return True
