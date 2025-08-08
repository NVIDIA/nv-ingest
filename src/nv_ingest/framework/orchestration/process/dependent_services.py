# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dependent services management for pipeline orchestration.

This module contains utilities for starting and managing dependent services
that the pipeline requires, such as message brokers and other infrastructure.
"""

import logging
import multiprocessing
import socket
from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker

logger = logging.getLogger(__name__)


def start_simple_message_broker(broker_client: dict) -> multiprocessing.Process:
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

    def broker_server():
        # Use max_queue_size from broker_params or default to 10000.
        broker_params = broker_client.get("broker_params", {})
        max_queue_size = broker_params.get("max_queue_size", 10000)
        server_host = broker_client.get("host", "0.0.0.0")
        server_port = broker_client.get("port", 7671)
        # Optionally, set socket options here for reuse.
        server = SimpleMessageBroker(server_host, server_port, max_queue_size)
        # Enable address reuse on the server socket.
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.serve_forever()

    p = multiprocessing.Process(target=broker_server)
    p.daemon = False
    p.start()
    logger.info(f"Started SimpleMessageBroker server in separate process on port {broker_client.get('port', 7671)}")

    return p
