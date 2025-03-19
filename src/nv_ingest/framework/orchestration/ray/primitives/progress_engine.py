# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import time
from typing import Dict, Any

import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage

logger = logging.getLogger(__name__)


@ray.remote
class ProgressEngine(RayActorSourceStage):
    """
    A progress engine actor source that continuously generates progress messages
    and writes them to its output edge. As a source stage, it does not support an input edge.
    """

    def __init__(self) -> None:
        self.running = False
        self.downstream_edge = None
        self.stats: Dict[str, int] = {"processed": 0}
        self.start_time = None

    @ray.method(num_returns=1)
    def start(self) -> bool:
        """
        Start the progress engine. This launches a background thread that continuously
        generates progress messages and writes them to the output edge.
        """
        if self.running:
            return False
        self.running = True
        self.start_time = time.time()
        # Launch the asynchronous loop in a background thread.
        threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True).start()
        return True

    async def _run(self) -> None:
        """
        Internal asynchronous loop that continuously generates progress messages and writes them
        to the downstream edge (if available).
        """
        while self.running:
            try:
                progress_message = await self._generate_progress()
                if progress_message is None:
                    await asyncio.sleep(0.1)
                    continue
                if self.downstream_edge:
                    # Write the progress message to the output edge.
                    await self.downstream_edge.write.remote(progress_message)
                self.stats["processed"] += 1
            except Exception as e:
                logger.exception(f"Error in ProgressEngine: {e}")
                await asyncio.sleep(0.1)

    async def _generate_progress(self) -> dict:
        """
        Generate a progress message. In this example, a dummy progress dictionary is returned.
        Replace this with your actual progress-generation logic as needed.
        """
        await asyncio.sleep(0.1)
        return {"progress": "dummy", "timestamp": time.time()}

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        """
        Stop the progress engine.
        """
        self.running = False
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        """
        Return statistics for the progress engine, including the number of processed messages
        and the elapsed time since starting.
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {"processed": self.stats["processed"], "elapsed": elapsed}

    @ray.method(num_returns=1)
    def set_output_edge(self, edge_handle: Any) -> bool:
        """
        Set the output edge (destination) for progress messages.
        """
        self.downstream_edge = edge_handle
        return True

    @ray.method(num_returns=1)
    def set_input_edge(self, edge_handle: Any) -> bool:
        """
        Source stages do not support an input edge.
        """
        raise NotImplementedError("Source stages do not support an input edge.")
