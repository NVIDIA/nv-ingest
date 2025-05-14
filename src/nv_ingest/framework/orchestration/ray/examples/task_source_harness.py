# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ray
import logging
import time

# Import the new source stage and its configuration
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    MessageBrokerTaskSourceStage,
    MessageBrokerTaskSourceConfig,
)


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RayTestHarness")

    # Define the Redis configuration for the MessageBrokerTaskSource
    redis_config = {
        "client_type": "redis",
        "host": "localhost",  # Adjust host if needed
        "port": 6379,  # Default Redis port
        "max_retries": 3,
        "max_backoff": 2,
        "connection_timeout": 5,
        "broker_params": {"db": 0, "use_ssl": False},
    }

    # Create an instance of the configuration for the source stage.
    config = MessageBrokerTaskSourceConfig(
        broker_client=redis_config,
        task_queue="ingest_task_queue",
        poll_interval=0.1,
    )

    message_broker_actor = MessageBrokerTaskSourceStage.remote(config)

    # Start the actor to begin fetching messages.
    ray.get(message_broker_actor.start.remote())
    logger.info("MessageBrokerTaskSource actor started. Listening for messages...")

    try:
        # Run indefinitely until a KeyboardInterrupt (Ctrl+C) is received.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Stopping actor...")
        ray.get(message_broker_actor.stop.remote())
        stats = ray.get(message_broker_actor.get_stats.remote())
        logger.info(f"Actor processing stats: {stats}")
    finally:
        ray.shutdown()
        logger.info("Ray shutdown complete.")


if __name__ == "__main__":
    main()
