import time

import ray
import logging

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting full pipeline test.")

    # Import the remote source actor.
    from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import MessageBrokerTaskSource

    # Define a simple sink actor that tracks throughput.
    @ray.remote
    class ThroughputSink:
        def __init__(self, **config):
            self.count = 0

        async def process(self, control_message):
            self.count += 1
            if self.count % 10 == 0:
                print(f"Sink processed {self.count} messages.")
            return

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
    pipeline.add_sink(name="sink", sink_actor=ThroughputSink, progress_engines=1)
    pipeline.make_edge("source", "sink", queue_size=100)
    pipeline.build()
    pipeline.start()

    logger.info("Full pipeline started. Source should pull messages from Redis and sink should process them.")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
