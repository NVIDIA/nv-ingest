# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ray
import logging
import time

# Import the source and sink stages and their configuration models.
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    MessageBrokerTaskSourceStage,
    MessageBrokerTaskSourceConfig,
)
from nv_ingest.framework.orchestration.ray.stages.sinks.message_broker_task_sink import (
    MessageBrokerTaskSinkStage,
    MessageBrokerTaskSinkConfig,
)

# Import the async queue edge.
from nv_ingest.framework.orchestration.ray.edges.async_queue_edge import AsyncQueueEdge


def main():
    # Initialize Ray.
    ray.init(ignore_reinit_error=True)

    # Set up basic logging.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RayPipelineHarness")

    # Define the Redis configuration for the message broker (used for both source and sink).
    redis_config = {
        "client_type": "redis",
        "host": "localhost",  # Adjust as needed.
        "port": 6379,
        "max_retries": 3,
        "max_backoff": 2,
        "connection_timeout": 5,
        "broker_params": {"db": 0, "use_ssl": False},
    }

    # Create a configuration instance for the source stage.
    source_config = MessageBrokerTaskSourceConfig(
        broker_client=redis_config,
        task_queue="ingest_task_queue",
        poll_interval=0.1,
        batch_size=10,
    )

    # Create a configuration instance for the sink stage.
    sink_config = MessageBrokerTaskSinkConfig(
        broker_client=redis_config,
        poll_interval=0.1,  # Using the same poll_interval; adjust as needed.
    )

    # Create an instance of the AsyncQueueEdge actor with a maximum size of 100.
    queue_edge = AsyncQueueEdge.remote(max_size=100, multi_reader=True, multi_writer=True)

    # Create an instance of the MessageBrokerTaskSourceStage actor.
    source_actor = MessageBrokerTaskSourceStage.remote(source_config, 1)

    # Create an instance of the MessageBrokerTaskSinkStage actor.
    sink_actor = MessageBrokerTaskSinkStage.remote(sink_config, 1)

    # Connect the stages:
    # The source's output edge is the queue_edge.
    ray.get(source_actor.set_output_edge.remote(queue_edge))
    # The sink's input edge is the same queue_edge.
    ray.get(sink_actor.set_input_edge.remote(queue_edge))

    # Start both actors.
    ray.get(source_actor.start.remote())
    ray.get(sink_actor.start.remote())
    logger.info("Source and Sink actors started, connected via AsyncQueueEdge.")

    try:
        # Run indefinitely until a KeyboardInterrupt (Ctrl+C) is received.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping actors...")
        ray.get(source_actor.stop.remote())
        ray.get(sink_actor.stop.remote())
        source_stats = ray.get(source_actor.get_stats.remote())
        sink_stats = ray.get(sink_actor.get_stats.remote())
        logger.info(f"Source stats: {source_stats}")
        logger.info(f"Sink stats: {sink_stats}")
    finally:
        ray.shutdown()
        logger.info("Ray shutdown complete.")


if __name__ == "__main__":
    main()
