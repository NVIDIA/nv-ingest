#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the Python orchestration framework no-op pipeline.

This script demonstrates:
1. Starting a SimpleBroker
2. Creating a source and sink
3. Running a no-op pipeline that reads from the broker and immediately returns the result
4. Submitting test messages and verifying they pass through
"""

import json
import logging
import time
from typing import Dict, Any

from .broker_starter import start_simple_message_broker
from src.nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
    PythonMessageBrokerTaskSourceConfig,
    SimpleClientConfig,
)
from src.nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
    PythonMessageBrokerTaskSinkConfig,
)
from .pipeline import PythonPipeline

from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_job(job_id: str = "test_job_001") -> Dict[str, Any]:
    """Create a test job for the pipeline."""
    return {
        "job_id": job_id,
        "job_type": "test",
        "payload": {
            "content": f"Test content for {job_id}",
            "metadata": {"source": "test_script", "timestamp": time.time()},
        },
        "response_channel": "test_response_queue",
    }


def submit_test_message(client: SimpleClient, queue_name: str, job: Dict[str, Any]) -> bool:
    """Submit a test message to the broker."""
    try:
        response = client.submit_message(queue_name, json.dumps(job))
        if response.response_code == 200:
            logger.info(f"Successfully submitted job {job['job_id']} to {queue_name}")
            return True
        else:
            logger.error(f"Failed to submit job, response code: {response.response_code}")
            return False
    except Exception as e:
        logger.error(f"Failed to submit test message: {e}")
        return False


def check_response_queue(client: SimpleClient, queue_name: str, timeout: int = 10) -> bool:
    """Check if there are responses in the response queue."""
    try:
        response = client.fetch_message(queue_name, timeout=(timeout, None))
        if response.response_code == 200 and response.response:
            logger.info(f"Found response in {queue_name}: {response.response}")
            return True
        elif response.response_code == 204:
            logger.info(f"No messages in {queue_name}")
            return False
        else:
            logger.warning(f"Unexpected response code: {response.response_code}")
            return False
    except Exception as e:
        logger.error(f"Failed to check response queue: {e}")
        return False


def test_noop_pipeline():
    """Test the no-op pipeline functionality."""
    logger.info("Starting no-op pipeline test")

    # Configuration
    broker_host = "localhost"
    broker_port = 8080
    task_queue = "test_task_queue"
    response_queue = "test_response_queue"

    # Step 1: Start SimpleBroker
    logger.info("Step 1: Starting SimpleBroker")
    broker_config = {"host": broker_host, "port": broker_port, "broker_params": {"max_queue_size": 1000}}

    broker_process = start_simple_message_broker(broker_config)
    time.sleep(2)  # Give broker time to start

    try:
        # Step 2: Create client for submitting test messages
        logger.info("Step 2: Creating test client")
        test_client = SimpleClient(broker_host, broker_port)

        # Step 3: Create source and sink configurations
        logger.info("Step 3: Creating source and sink")

        source_config = PythonMessageBrokerTaskSourceConfig(
            broker_client=SimpleClientConfig(client_type="simple", host=broker_host, port=broker_port),
            task_queue=task_queue,
            poll_interval=0.1,
        )

        sink_config = PythonMessageBrokerTaskSinkConfig(
            broker_client=SimpleClientConfig(client_type="simple", host=broker_host, port=broker_port),
            poll_interval=0.1,
        )

        source = PythonMessageBrokerTaskSource(source_config)
        sink = PythonMessageBrokerTaskSink(sink_config)

        # Step 4: Create no-op pipeline (no processing functions)
        logger.info("Step 4: Creating no-op pipeline")
        pipeline = PythonPipeline(source, sink, processing_functions=[])

        # Step 5: Submit test messages
        logger.info("Step 5: Submitting test messages")
        test_jobs = [create_test_job("test_job_001"), create_test_job("test_job_002"), create_test_job("test_job_003")]

        for job in test_jobs:
            success = submit_test_message(test_client, task_queue, job)
            if not success:
                logger.error(f"Failed to submit job {job['job_id']}")
                return False

        # Step 6: Run pipeline for a limited number of iterations
        logger.info("Step 6: Running no-op pipeline")
        pipeline.run(max_iterations=10, poll_interval=0.5)

        # Step 7: Check results
        logger.info("Step 7: Checking pipeline statistics")
        stats = pipeline.get_stats()
        logger.info(f"Pipeline stats: {stats}")

        # Step 8: Check response queue for processed messages
        logger.info("Step 8: Checking response queue")
        responses_found = 0
        for i in range(5):  # Check up to 5 times
            if check_response_queue(test_client, response_queue, timeout=2):
                responses_found += 1
            time.sleep(1)

        logger.info(f"Found {responses_found} responses in response queue")

        # Evaluate test results
        if stats["processed_count"] > 0:
            logger.info("✅ SUCCESS: No-op pipeline processed messages successfully!")
            logger.info(f"Processed {stats['processed_count']} messages with {stats['error_count']} errors")
            return True
        else:
            logger.error("❌ FAILURE: No messages were processed by the pipeline")
            return False

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False

    finally:
        # Cleanup: Stop broker
        logger.info("Cleaning up: Stopping SimpleBroker")
        if broker_process and broker_process.is_alive():
            broker_process.terminate()
            broker_process.join(timeout=5)
            if broker_process.is_alive():
                broker_process.kill()


if __name__ == "__main__":
    success = test_noop_pipeline()
    exit(0 if success else 1)
