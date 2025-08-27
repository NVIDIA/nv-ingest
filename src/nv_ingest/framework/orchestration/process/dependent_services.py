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
import threading
import time
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

    Notes
    -----
    If the environment variable `NV_INGEST_BROKER_SKIP_IF_PORT_OPEN` is set to "1"
    and the target port is already in use, this function will log and return None
    instead of spawning a new process. This provides an opt-in idempotency guard
    to avoid noisy spawn attempts when a broker is already running.
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

    skip_if_open = os.environ.get("NV_INGEST_BROKER_SKIP_IF_PORT_OPEN") == "1"
    if _is_port_open(server_host, server_port):
        if skip_if_open:
            logger.info(
                f"SimpleMessageBroker port already in use at {server_host}:{server_port}; "
                f"skipping spawn due to NV_INGEST_BROKER_SKIP_IF_PORT_OPEN=1"
            )
            return None
        logger.warning(
            f"SimpleMessageBroker port already in use at {server_host}:{server_port}; "
            f"continuing to spawn a broker process (tests expect a Process to be returned)"
        )

    def broker_server():
        # Optionally, set socket options here for reuse (note: binding occurs in server __init__).
        server = SimpleMessageBroker(server_host, server_port, max_queue_size)
        try:
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        server.serve_forever()

    p = multiprocessing.Process(target=broker_server)
    # If we're launching from inside the pipeline subprocess, mark daemon so the
    # broker dies automatically when the subprocess exits.
    p.daemon = os.environ.get("NV_INGEST_BROKER_IN_SUBPROCESS") == "1"
    p.start()
    logger.info(f"Started SimpleMessageBroker server in separate process on port {server_port}")

    return p


def start_simple_message_broker_inthread(broker_client: dict):
    """
    Starts a SimpleMessageBroker server in a background daemon thread (same process).

    Parameters
    ----------
    broker_client : dict
        Broker configuration. Expected keys include:
          - "host": host to bind the server to (default: 0.0.0.0)
          - "port": port to bind the server to (default: 7671)
          - "broker_params": optionally including "max_queue_size"

    Returns
    -------
    dict
        A handle dict containing {"server": server, "thread": thread, "host": host, "port": port}.

    Notes
    -----
    - Honors NV_INGEST_BROKER_SKIP_IF_PORT_OPEN=1 similarly to the process-based starter.
    - The returned server is the in-process singleton, enabling in-process client detection.
    """

    broker_params = broker_client.get("broker_params", {})
    max_queue_size = broker_params.get("max_queue_size", 10000)
    server_host = broker_client.get("host", "0.0.0.0")
    server_port = broker_client.get("port", 7671)

    def _is_port_open(host: str, port: int) -> bool:
        check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        try:
            with socket.create_connection((check_host, port), timeout=0.5):
                return True
        except Exception:
            return False

    skip_if_open = os.environ.get("NV_INGEST_BROKER_SKIP_IF_PORT_OPEN") == "1"
    if _is_port_open(server_host, server_port):
        if skip_if_open:
            logger.info(
                f"SimpleMessageBroker port already in use at {server_host}:{server_port}; "
                f"skipping in-thread start due to NV_INGEST_BROKER_SKIP_IF_PORT_OPEN=1"
            )
            return None
        logger.warning(
            f"SimpleMessageBroker port already in use at {server_host}:{server_port}; "
            f"continuing to start a broker thread (tests may expect a handle to be returned)"
        )

    server = SimpleMessageBroker(server_host, server_port, max_queue_size)

    def run_broker():
        try:
            server.serve_forever()
        except Exception as e:
            logger.error(f"SimpleMessageBroker thread terminated with error: {e}")

    t = threading.Thread(target=run_broker, name="SimpleMessageBrokerThread", daemon=True)
    t.start()
    # Give the server a brief moment to bind/listen
    time.sleep(0.05)
    logger.info(f"Started SimpleMessageBroker in background thread on {server_host}:{server_port}")
    return {"server": server, "thread": t, "host": server_host, "port": server_port}
