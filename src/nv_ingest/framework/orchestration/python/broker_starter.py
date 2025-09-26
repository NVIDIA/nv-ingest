# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
from typing import Dict, Any

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker

logger = logging.getLogger(__name__)


def start_simple_message_broker(broker_client: Dict[str, Any]) -> multiprocessing.Process:
    """
    Starts a SimpleMessageBroker server in a separate process.

    Parameters
    ----------
    broker_client : dict
        Broker configuration. Expected keys include:
          - "port": the port to bind the server to,
          - "broker_params": optionally including "max_queue_size",
          - and any other parameters required by SimpleMessageBroker.

    Returns
    -------
    multiprocessing.Process
        The process running the SimpleMessageBroker server.
    """
    host = broker_client.get("host", "0.0.0.0")
    port = broker_client.get("port", 7671)
    broker_params = broker_client.get("broker_params", {})
    max_queue_size = broker_params.get("max_queue_size", 1000)

    logger.info(f"Starting SimpleMessageBroker on {host}:{port} with max_queue_size={max_queue_size}")

    def _run_broker():
        """Internal function to run the broker in a separate process."""
        try:
            broker = SimpleMessageBroker(host, port, max_queue_size)
            logger.info(f"SimpleMessageBroker started successfully on {host}:{port}")
            broker.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start SimpleMessageBroker: {e}")
            raise

    # Start the broker in a separate process
    broker_process = multiprocessing.Process(target=_run_broker)
    broker_process.start()

    logger.info(f"SimpleMessageBroker process started with PID: {broker_process.pid}")
    return broker_process
