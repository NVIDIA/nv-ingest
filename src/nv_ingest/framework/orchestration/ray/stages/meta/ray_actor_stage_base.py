# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from abc import ABC, abstractmethod

import ray
from pydantic import BaseModel
from typing import Any
import logging

logger = logging.getLogger(__name__)


class RayActorStage(ABC):
    def __init__(self, config: BaseModel, progress_engine_count: int = 1) -> None:
        self.config = config
        self.progress_engine_count = progress_engine_count
        self.input_edge = None  # Used for non-source stages.
        self.output_edge = None
        self.running = False
        self.stats = {"processed": 0}
        self.start_time = None

    def read_input(self) -> Any:
        if self.input_edge is None:
            raise ValueError("Input edge not set")
        # Call the remote method synchronously.
        return ray.get(self.input_edge.read.remote())

    @abstractmethod
    def on_data(self, control_message: Any) -> Any:
        """
        Process an incoming control message and return an updated control message.
        Must be implemented by subclasses.
        """
        pass

    def _processing_loop(self) -> None:
        while self.running:
            try:
                control_message = self.read_input()
                if control_message is None:
                    time.sleep(self.config.poll_interval)
                    continue
                updated_cm = self.on_data(control_message)
                if updated_cm and self.output_edge:
                    # Synchronously call write on the output edge.
                    ray.get(self.output_edge.write.remote(updated_cm))
                self.stats["processed"] += 1
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                time.sleep(self.config.poll_interval)

    @ray.method(num_returns=1)
    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.start_time = time.time()
        threading.Thread(target=self._processing_loop, daemon=True).start()
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        self.running = False
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {"processed": self.stats["processed"], "elapsed": elapsed}

    @ray.method(num_returns=1)
    def set_input_edge(self, edge_handle: Any) -> bool:
        self.input_edge = edge_handle
        return True

    @ray.method(num_returns=1)
    def set_output_edge(self, edge_handle: Any) -> bool:
        self.output_edge = edge_handle
        return True
