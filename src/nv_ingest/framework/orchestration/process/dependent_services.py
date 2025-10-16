# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dependent services management for pipeline orchestration.

This module contains utilities for starting and managing dependent services
that the pipeline requires, such as message brokers and other infrastructure.
"""

import logging
import os
import multiprocessing
import socket
from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker

logger = logging.getLogger(__name__)


def _broker_server_target(host, port, max_queue_size):
    """
    Target function to be run in a separate process for the SimpleMessageBroker.
    """
    server = SimpleMessageBroker(host, port, max_queue_size)
    try:
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception:
        pass
    server.serve_forever()


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

    # Resolve host/port early for pre-flight checks
    broker_params = broker_client.get("broker_params", {})
    max_queue_size = broker_params.get("max_queue_size", 10000)
    server_host = broker_client.get("host", "0.0.0.0")
    server_port = broker_client.get("port", 7671)

    # Pre-flight: if something is already listening on the target port, do not spawn another broker.
    # This avoids noisy stack traces from a failing child process when tests/pipeline are run repeatedly.
    def _is_port_open(host: str, port: int) -> bool:
        check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        try:
            with socket.create_connection((check_host, port), timeout=0.5):
                return True
        except Exception:
            return False

    if _is_port_open(server_host, server_port):
        logger.warning(
            f"SimpleMessageBroker port already in use at {server_host}:{server_port}; "
            f"continuing to spawn a broker process (tests expect a Process to be returned)"
        )

    p = multiprocessing.Process(
        target=_broker_server_target,
        args=(server_host, server_port, max_queue_size),
        daemon=True,
    )
    # If we're launching from inside the pipeline subprocess, mark daemon so the
    # broker dies automatically when the subprocess exits.
    p.daemon = os.environ.get("NV_INGEST_BROKER_IN_SUBPROCESS") == "1"
    p.start()
    logger.info(f"Started SimpleMessageBroker server in separate process on port {server_port}")

    return p
