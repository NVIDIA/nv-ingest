# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import ray

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector import MetadataInjectionStage
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import MessageBrokerTaskSource

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


# Define a dummy processor stage for the downstream stage.
@ray.remote
class DummyProcessor2:
    def __init__(self, **config):
        self.count = 0
        self.downstream_queue = None

    async def process(self, control_message: Dict[str, Any]) -> Dict[str, Any]:
        self.count += 1
        print(f"[{ray.get_runtime_context().get_node_id()}] DummyProcessor2 processed {self.count} messages.")
        await asyncio.sleep(0.05)
        if self.downstream_queue:
            await self.downstream_queue.put.remote(control_message)
        return control_message

    def set_output_queue(self, queue_handle: Any) -> bool:
        self.downstream_queue = queue_handle
        return True


@ray.remote
class OutputValidator:
    """
    A Ray actor that validates the structure of IngestControlMessage objects.

    It checks that the payload is a pandas DataFrame and that required metadata keys exist.
    If validation passes, it prints a confirmation message; if not, it marks the message as failed.

    Attributes
    ----------
    count : int
        Number of messages processed.
    downstream_queue : Any, optional
        The Ray actor handle for the downstream queue.
    """

    def __init__(self, **config: Any) -> None:
        self.count: int = 0
        self.downstream_queue: Any = None

    async def process(self, control_message: IngestControlMessage) -> IngestControlMessage:
        self.count += 1
        try:
            df = control_message.payload()
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Payload is not a pandas DataFrame")
            metadata = getattr(control_message, "metadata", {})
            required_keys = ["job_id", "response_channel", "timestamp"]
            for key in required_keys:
                if key not in metadata:
                    raise ValueError(f"Missing required metadata key: {key}:\n{json.dumps(metadata, indent=2)}")
            print(
                f"[{ray.get_runtime_context().get_node_id()}] "
                f"OutputValidator processed {self.count} messages - validation passed."
            )
        except Exception as e:
            print(f"Output validation error: {e}")
            control_message.set_metadata("output_validation", "failed")
        if self.downstream_queue:
            await self.downstream_queue.put.remote(control_message)
        return control_message

    def set_output_queue(self, queue_handle: Any) -> bool:
        """
        Set the downstream queue for this validator.

        Parameters
        ----------
        queue_handle : Any
            The Ray actor handle representing the downstream queue.

        Returns
        -------
        bool
            True if the downstream queue was set successfully.
        """
        self.downstream_queue = queue_handle
        return True


@ray.remote
class ThroughputSink:
    def __init__(self, **config):
        self.count = 0

    async def process(self, control_message: Dict[str, Any]) -> Dict[str, Any]:
        self.count += 1
        if self.count % 10 == 0:
            print(f"Sink processed {self.count} messages.")
        return control_message

    def set_output_queue(self, queue_handle: Any) -> bool:
        return True


# Redis configuration for the source.
redis_config: Dict[str, Any] = {
    "client_type": "redis",
    "host": "localhost",
    "port": 6379,
    "max_retries": 3,
    "max_backoff": 2,
    "connection_timeout": 5,
    "broker_params": {"db": 0, "use_ssl": False},
}

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting multi-stage pipeline test.")

    # Build the pipeline.
    pipeline = RayPipeline()
    pipeline.add_source(
        name="source",
        source_actor=MessageBrokerTaskSource,
        broker_client=redis_config,
        task_queue="morpheus_task_queue",
        progress_engines=1,
        poll_interval=0.1,
        batch_size=10,
    )
    # Use MetadataInjectionStage in place of DummyProcessor.
    pipeline.add_stage(
        name="metadata_injection",
        stage_actor=MetadataInjectionStage,
        progress_engines=1,
    )
    pipeline.add_stage(
        name="validator",
        stage_actor=OutputValidator,
        progress_engines=1,
    )
    pipeline.add_stage(
        name="processor2",
        stage_actor=DummyProcessor2,
        progress_engines=1,
    )
    pipeline.add_sink(
        name="sink",
        sink_actor=ThroughputSink,
        progress_engines=1,
    )
    # Wire the stages: source → metadata_injection → validator → processor2 → sink.
    pipeline.make_edge("source", "metadata_injection", queue_size=100)
    pipeline.make_edge("metadata_injection", "validator", queue_size=100)
    pipeline.make_edge("validator", "processor2", queue_size=100)
    pipeline.make_edge("processor2", "sink", queue_size=100)

    pipeline.build()

    # Visualize the pipeline as a text graph.
    pipeline.visualize(mode="text", verbose=True, max_width=120)

    pipeline.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
