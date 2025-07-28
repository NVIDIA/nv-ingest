#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python pipeline harness for the Python orchestration framework.

This script provides a harness to:
1. Optionally start a SimpleBroker in-process (for testing)
2. Create source and sink components
3. Set up a pipeline with metadata injection using the new interface
4. Run the pipeline in background

Note: In production/container environments, the broker should run externally.
"""

import logging
import time
import threading

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker
from nv_ingest.framework.orchestration.python.python_pipeline import PythonPipeline
from nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
    PythonMessageBrokerTaskSourceConfig,
    SimpleClientConfig as SourceSimpleClientConfig,
)
from nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
    PythonMessageBrokerTaskSinkConfig,
    SimpleClientConfig as SinkSimpleClientConfig,
)
from nv_ingest.framework.orchestration.python.stages.injectors.metadata_injector import (
    PythonMetadataInjectionStage,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def start_broker_in_process(host: str = "localhost", port: int = 7671, max_queue_size: int = 1000) -> threading.Thread:
    """
    Start SimpleBroker in a background thread for testing purposes.

    In production/container environments, the broker should run externally.
    """

    def run_broker():
        try:
            broker = SimpleMessageBroker(host=host, port=port, max_queue_size=max_queue_size)
            logger.info(f"Starting SimpleBroker on {host}:{port} with max_queue_size={max_queue_size}")
            broker.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start broker: {e}")

    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    return broker_thread


def main():
    """Main harness function."""
    logger.info("Starting Python pipeline harness")

    # Configuration
    broker_host = "0.0.0.0"
    broker_port = 7671
    task_queue = "ingest_task_queue"  # Match the queue name used by nv_ingest_client
    start_local_broker = True  # Set to False if broker runs externally

    broker_thread = None

    # Step 1: Optionally start SimpleBroker in-process (for testing only)
    if start_local_broker:
        logger.info("Starting SimpleBroker in-process (testing mode)...")
        broker_thread = start_broker_in_process(broker_host, broker_port, max_queue_size=1000)
        time.sleep(2)  # Give broker time to start
    else:
        logger.info(f"Assuming external broker running on {broker_host}:{broker_port}")

    try:
        # Step 2: Create source configuration
        logger.info("Creating source configuration...")
        source_config = PythonMessageBrokerTaskSourceConfig(
            broker_client=SourceSimpleClientConfig(host=broker_host, port=broker_port),
            task_queue=task_queue,
            poll_interval=0.1,
        )

        # Step 3: Create sink configuration
        logger.info("Creating sink configuration...")
        sink_config = PythonMessageBrokerTaskSinkConfig(
            broker_client=SinkSimpleClientConfig(host=broker_host, port=broker_port), poll_interval=0.1
        )

        # Step 4: Create source and sink instances
        logger.info("Creating source and sink instances...")
        source = PythonMessageBrokerTaskSource(source_config)
        sink = PythonMessageBrokerTaskSink(sink_config)

        # Step 4.5: Test broker connection before starting pipeline
        logger.info("Testing broker connection...")
        try:
            # Test basic broker connectivity
            test_client = SourceSimpleClientConfig(host=broker_host, port=broker_port)
            from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient

            test_simple_client = SimpleClient(
                host=test_client.host,
                port=test_client.port,
                max_retries=test_client.max_retries,
                max_backoff=test_client.max_backoff,
                connection_timeout=test_client.connection_timeout,
            )

            # Test queue size to verify connection
            size_response = test_simple_client.size(task_queue)
            logger.info(
                f"Broker connection test - Queue '{task_queue}' "
                f"size: {size_response.response_code} - {size_response.response}"
            )

            if size_response.response_code == 0:
                logger.info(f" Broker connection successful. Queue size: {size_response.response}")
            else:
                logger.error(
                    f" Broker connection failed. Response: {size_response.response_code} -"
                    f" {size_response.response_reason}"
                )

            # Test message submission and retrieval
            logger.info("Testing message submission and retrieval...")
            test_message = '{"job_id": "test-123", "test": "data"}'

            # Submit test message
            submit_response = test_simple_client.submit_message(task_queue, test_message)
            logger.info(f"Test message submit: {submit_response.response_code} - {submit_response.response}")

            if submit_response.response_code == 0:
                # Check queue size after submission
                size_after = test_simple_client.size(task_queue)
                logger.info(f"Queue size after test message: {size_after.response}")

                # Try to fetch the message
                fetch_response = test_simple_client.fetch_message(task_queue, timeout=(5, None))
                logger.info(f"Test message fetch: {fetch_response.response_code} - {fetch_response.response[:100]}")

                if fetch_response.response_code == 0:
                    logger.info("✓ Message submission and retrieval test successful")
                else:
                    logger.error(f"✗ Message fetch failed: {fetch_response.response_reason}")
            else:
                logger.error(f"✗ Message submission failed: {submit_response.response_reason}")

        except Exception as e:
            logger.error(f"✗ Broker connection test failed: {e}")

        # Step 5: Create pipeline using new interface
        logger.info("Creating pipeline with metadata injection...")
        pipeline = PythonPipeline()

        # Add source using new interface
        pipeline.add_source(name="message_broker_source", source_actor=source, config=source_config)

        # Add metadata injector stage
        metadata_injector_config = {}
        metadata_injector = PythonMetadataInjectionStage(metadata_injector_config)
        pipeline.add_stage(name="metadata_injector", stage_actor=metadata_injector, config=metadata_injector_config)

        # Add sink using new interface
        pipeline.add_sink(name="message_broker_sink", sink_actor=sink, config=sink_config)

        logger.info("Pipeline created with source → metadata_injector → sink")

        # Step 6: Start pipeline in background
        logger.info("Starting pipeline...")
        pipeline.start()

        logger.info("Pipeline is now running in background. Submit messages to the task queue to see processing.")
        logger.info(f"Task queue: {task_queue}")
        logger.info("Press Ctrl+C to stop the pipeline.")

        # Keep main thread alive and periodically show stats
        try:
            while True:
                time.sleep(10)
                stats = pipeline.get_stats()
                logger.info(
                    f"Pipeline stats: processed={stats['processed_count']}, "
                    f"errors={stats['error_count']}, "
                    f"rate={stats['processing_rate_cps']:.2f} msg/sec"
                )
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")

    except Exception as e:
        logger.error(f"Pipeline harness failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")

        # Stop pipeline if it exists
        if "pipeline" in locals():
            pipeline.stop()

        # Note: In-process broker thread will stop when main process exits (daemon=True)
        if broker_thread and broker_thread.is_alive():
            logger.info("In-process broker will stop with main process")


if __name__ == "__main__":
    main()
