# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import ray
import logging

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline

logger = logging.getLogger(__name__)

###############################################
# Multi-stage pipeline test harness
###############################################
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting multi-stage pipeline test.")

    # Import the remote source actor.
    from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import MessageBrokerTaskSource

    # Define a dummy processing stage that logs, forwards the message, and prints its own output.
    @ray.remote
    class DummyProcessor:
        def __init__(self, **config):
            self.count = 0
            self.downstream_queue = None

        async def process(self, control_message):
            self.count += 1
            print(f"[{ray.get_runtime_context().get_node_id()}] DummyProcessor processed {self.count} messages.")
            await asyncio.sleep(0.05)
            if self.downstream_queue:
                # Forward the message to the downstream queue.
                await self.downstream_queue.put.remote(control_message)
            return control_message

        def set_output_queue(self, queue_handle):
            self.downstream_queue = queue_handle
            return True

    @ray.remote
    class DummyProcessor2:
        def __init__(self, **config):
            self.count = 0
            self.downstream_queue = None

        async def process(self, control_message):
            self.count += 1
            print(f"[{ray.get_runtime_context().get_node_id()}] DummyProcessor2 processed {self.count} messages.")
            await asyncio.sleep(0.05)
            if self.downstream_queue:
                # Forward the message to the downstream queue.
                await self.downstream_queue.put.remote(control_message)
            return control_message

        def set_output_queue(self, queue_handle):
            self.downstream_queue = queue_handle
            return True

    # Define a simple sink actor that tracks throughput.
    @ray.remote
    class ThroughputSink:
        def __init__(self, **config):
            self.count = 0

        async def process(self, control_message):
            self.count += 1
            if self.count % 10 == 0:
                print(f"Sink processed {self.count} messages.")
            return control_message

        def set_output_queue(self, queue_handle):
            # Not used by the sink.
            return True

    # Redis configuration for the source.
    redis_config = {
        "client_type": "redis",
        "host": "localhost",
        "port": 6379,
        "max_retries": 3,
        "max_backoff": 2,
        "connection_timeout": 5,
        "broker_params": {"db": 0, "use_ssl": False},
    }

    # Build the pipeline:
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
    pipeline.add_stage(
        name="processor1",
        stage_actor=DummyProcessor,
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
    # Wire the stages: source -> processor1 -> processor2 -> sink.
    pipeline.make_edge("source", "processor1", queue_size=100)
    pipeline.make_edge("processor1", "processor2", queue_size=100)
    pipeline.make_edge("processor2", "sink", queue_size=100)

    pipeline.build()
    pipeline.start()

    logger.info("Multi-stage pipeline started. Flow: Source -> Processor1 -> Processor2 -> Sink.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
