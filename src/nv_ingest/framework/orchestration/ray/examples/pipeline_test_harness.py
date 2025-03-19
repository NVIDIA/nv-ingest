# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ray
import logging
import time
from typing import Dict, Any

# Import our new pipeline class and AsyncQueueEdge.
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector import MetadataInjectionStage
from nv_ingest.framework.orchestration.ray.stages.sinks.message_broker_task_sink import (
    MessageBrokerTaskSinkStage,
    MessageBrokerTaskSinkConfig,
)

# Import our new stage implementations.
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    MessageBrokerTaskSourceStage,
    MessageBrokerTaskSourceConfig,
)
from nv_ingest.framework.orchestration.ray.stages.utility.throughput_monitor import ThroughputMonitorStage

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
    logger = logging.getLogger("RayPipelineHarness")
    logger.info("Starting multi-stage pipeline test.")

    # Build the pipeline.
    pipeline = RayPipeline()

    # Create configuration instances for the source and sink stages.
    source_config = MessageBrokerTaskSourceConfig(
        broker_client=redis_config,
        task_queue="morpheus_task_queue",
        poll_interval=0.1,
    )
    sink_config = MessageBrokerTaskSinkConfig(
        broker_client=redis_config,
        poll_interval=0.1,
    )

    # Add stages:
    # 1. Source stage.
    pipeline.add_source(
        name="source",
        source_actor=MessageBrokerTaskSourceStage,
        config=source_config,
        progress_engine_count=1,
    )
    # 2. Metadata injection stage.
    pipeline.add_stage(
        name="metadata_injection",
        stage_actor=MetadataInjectionStage,
        config={},  # Pass an empty config or a config specific to metadata injection if needed.
        progress_engine_count=1,
    )
    # 3. Throughput Monitor
    pipeline.add_stage(
        name="throughput_monitor",
        stage_actor=ThroughputMonitorStage,
        config={},
        progress_engine_count=1,
    )
    # 4. Sink stage.
    pipeline.add_sink(
        name="sink",
        sink_actor=MessageBrokerTaskSinkStage,
        config=sink_config,
        progress_engine_count=1,
    )

    # Wire the stages together via AsyncQueueEdges.
    # The intended flow is: source → metadata_injection → sink.
    pipeline.make_edge("source", "metadata_injection", queue_size=100)
    pipeline.make_edge("metadata_injection", "throughput_monitor", queue_size=100)
    pipeline.make_edge("throughput_monitor", "sink", queue_size=100)

    # Build the pipeline (this instantiates actors and wires edges).
    pipeline.build()

    # Optionally, visualize the pipeline graph.
    # pipeline.visualize(mode="text", verbose=True, max_width=120)

    # Start the pipeline.
    pipeline.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
        logger.info("Ray shutdown complete.")
