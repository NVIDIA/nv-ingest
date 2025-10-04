# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import threading
import time
from uuid import uuid4

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient, SimpleMessageBroker

HOST = "127.0.0.1"
MAX_QUEUE_SIZE = 10

PUSH_TIMEOUT_MSG = "PUSH operation timed out."


@pytest.fixture(scope="session")
def broker():
    """Start broker on ephemeral port and yield (server, host, port)."""
    server = SimpleMessageBroker(HOST, 0, MAX_QUEUE_SIZE)
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    # small wait to ensure bound
    time.sleep(0.2)
    try:
        yield server, host, port
    finally:
        # Best-effort shutdown without risking indefinite hang
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        # Avoid unbounded join in case shutdown misbehaves
        try:
            server_thread.join(timeout=2.0)
        except Exception:
            pass
        # Clear any process-wide singleton to avoid lingering references
        try:
            setattr(SimpleMessageBroker, "_instance", None)
        except Exception:
            pass


@pytest.fixture
def broker_addr(broker):
    """Return (host, port) for the running broker."""
    _, host, port = broker
    return host, port


@pytest.fixture
def client(broker_addr):
    """Fixture to provide a SimpleClient instance using explicit socket interface."""
    host, port = broker_addr
    return SimpleClient(host, port, connection_timeout=5, interface_type="socket")


def test_message_ordering(broker_addr):
    """Test that messages are popped in the same order they were pushed (FIFO)."""
    host, port = broker_addr
    client = SimpleClient(host, port, connection_timeout=5, interface_type="socket")
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(5)]

    # Push messages in order
    for msg in messages:
        response = client.submit_message(queue_name, msg)
        assert response.response_code == 0

    # Pop messages and verify order
    popped_messages = []
    for _ in messages:
        response = client.fetch_message(queue_name)
        assert response.response_code == 0
        popped_messages.append(response.response)

    assert popped_messages == messages, "Messages popped are not in the same order as they were pushed."


def test_push_to_full_queue(broker_addr):
    """Test pushing messages to a full queue."""
    host, port = broker_addr
    client = SimpleClient(host, port, connection_timeout=5, interface_type="socket")
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(MAX_QUEUE_SIZE)]

    # Push messages until the queue is full
    for msg in messages:
        response = client.submit_message(queue_name, msg, timeout=(1, None))
        assert response.response_code == 0

    # Attempt to push one more message beyond capacity
    response = client.submit_message(queue_name, "Extra Message", timeout=(1, None))
    assert response.response_code == 1, "Expected failure when pushing to a full queue."
    assert response.response_reason == PUSH_TIMEOUT_MSG


def test_invalid_inputs(broker_addr):
    """Test that the client handles invalid inputs properly."""
    host, port = broker_addr
    client = SimpleClient(host, port, connection_timeout=5, interface_type="socket")

    # Test with empty queue name
    response = client.submit_message("", "Test Message")
    assert response.response_code == 1
    assert response.response_reason == "Invalid queue name."

    # Test with empty message
    queue_name = f"test_queue_{uuid4()}"
    response = client.submit_message(queue_name, "")
    assert response.response_code == 1
    assert response.response_reason == "Invalid message."

    # Test with non-string queue name
    response = client.submit_message(12345, "Test Message")
    assert response.response_code == 1
    assert response.response_reason == "Invalid queue name."

    # Test with non-string message
    response = client.submit_message(queue_name, 12345)
    assert response.response_code == 1
    assert response.response_reason == "Invalid message."

    # Test with negative timeout
    response = client.submit_message(queue_name, "Test Message", timeout=(-5, None))
    assert response.response_code == 1
    assert PUSH_TIMEOUT_MSG in response.response_reason

    # Test with extremely large timeout
    response = client.submit_message(queue_name, "Test Message", timeout=(1e10, None))
    assert response.response_code == 1


def test_server_unavailable():
    """Test client's behavior when the server is unavailable."""
    client = SimpleClient(HOST, 9991, connection_timeout=2, interface_type="socket")
    queue_name = f"test_queue_{uuid4()}"

    # Do not start the broker_server fixture to simulate server unavailability

    response = client.fetch_message(queue_name, timeout=(1, None))
    assert response.response_code == 2
    assert "Job not ready." in response.response_reason


def test_operation_timeout(broker_addr):
    """Test client's behavior when an operation times out."""
    host, port = broker_addr
    client = SimpleClient(host, port, connection_timeout=2, interface_type="socket")
    queue_name = f"test_queue_{uuid4()}"

    # Fill the queue
    for i in range(MAX_QUEUE_SIZE):
        client.submit_message(queue_name, f"msg {i}", timeout=(1, None))

    # This push should timeout
    response = client.submit_message(queue_name, "Test Message", timeout=(1, None))
    assert response.response_code == 1
    assert PUSH_TIMEOUT_MSG in response.response_reason


def test_pop_success(client):
    """Test successful POP operation."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Push a message first
    push_response = client.submit_message(queue_name, message)
    assert push_response.response_code == 0

    # Pop the message
    pop_response = client.fetch_message(queue_name)
    assert pop_response.response_code == 0
    assert pop_response.response == message
    assert pop_response.transaction_id is not None


def test_size_command(client):
    """Test SIZE command."""
    queue_name = f"test_queue_{uuid4()}"
    messages = ["Message 1", "Message 2", "Message 3"]

    # Push messages
    for msg in messages:
        response = client.submit_message(queue_name, msg)
        assert response.response_code == 0

    # Check size
    size_response = client.size(queue_name)
    assert size_response.response_code == 0
    assert size_response.response == str(len(messages))


def test_push_large_message(broker_addr):
    """Test pushing a large message."""
    queue_name = f"test_queue_{uuid4()}"
    message = "A" * (5 * 1024 * 1024)  # 5MB message
    host, port = broker_addr
    client = SimpleClient(host, port, connection_timeout=10, interface_type="socket")
    response = client.submit_message(queue_name, message, timeout=(5, None))
    assert response.response_code == 0
    assert response.response == "Data stored."
    assert response.transaction_id is not None


def test_pop_empty_queue(client):
    """Test popping from an empty queue."""
    queue_name = f"test_queue_{uuid4()}"

    response = client.fetch_message(queue_name, timeout=(1, None))
    assert response.response_code == 2
    assert "Job not ready" in response.response_reason


def test_push_with_timeout(client):
    """Test PUSH operation with a timeout."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    response = client.submit_message(queue_name, message, timeout=(5, None))
    assert response.response_code == 0
    assert response.response == "Data stored."
    assert response.transaction_id is not None


def test_pop_with_timeout(client):
    """Test POP operation with a timeout."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Push a message first
    push_response = client.submit_message(queue_name, message)
    assert push_response.response_code == 0

    # Pop the message with a timeout
    pop_response = client.fetch_message(queue_name, timeout=(5, None))
    assert pop_response.response_code == 0
    assert pop_response.response == message
    assert pop_response.transaction_id is not None


def test_push_error_handling(client):
    """Test error handling during PUSH."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Simulate connection error by closing the client's socket
    client._socket = None

    response = client.submit_message(queue_name, message)
    # The client should handle the error and retry
    assert response.response_code == 0
    assert response.response == "Data stored."


def test_pop_error_handling(client):
    """Test error handling during POP."""
    queue_name = f"test_queue_{uuid4()}"

    # Simulate connection error by closing the client's socket
    client._socket = None

    response = client.fetch_message(queue_name, timeout=(1, None))
    # The client should handle the error and return an appropriate response
    assert response.response_code == 2
    assert "Job not ready." in response.response_reason


def test_multiple_clients(broker_addr):
    """Test multiple clients interacting with the server."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(5)]
    host, port = broker_addr
    clients = [SimpleClient(host, port, connection_timeout=5, interface_type="socket") for _ in range(5)]

    # Each client pushes a message
    for i, c in enumerate(clients):
        response = c.submit_message(queue_name, messages[i])
        assert response.response_code == 0

    # Each client pops a message
    popped_messages = []
    for c in clients:
        response = c.fetch_message(queue_name)
        assert response.response_code == 0
        popped_messages.append(response.response)

    # Verify that all messages were popped
    assert sorted(popped_messages) == sorted(messages)


def test_concurrent_push_pop(broker_addr):
    """Test concurrent PUSH and POP operations."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(10)]
    results = []
    lock = threading.Lock()
    host, port = broker_addr
    push_client = SimpleClient(host, port, connection_timeout=5, interface_type="socket")
    pop_client = SimpleClient(host, port, connection_timeout=5, interface_type="socket")

    def push(msg):
        response = push_client.submit_message(queue_name, msg)
        with lock:
            results.append(response)

    def pop():
        response = pop_client.fetch_message(queue_name)
        with lock:
            results.append(response)

    threads = []
    for msg in messages:
        t_push = threading.Thread(target=push, args=(msg,))
        t_pop = threading.Thread(target=pop)
        threads.extend([t_push, t_pop])

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify that all messages were pushed and popped
    pushed = [r for r in results if r.response == "Data stored."]
    popped = [r for r in results if r.response_code == 0 and r.response.startswith("Message")]
    assert len(pushed) == len(messages)
    assert len(popped) == len(messages)
