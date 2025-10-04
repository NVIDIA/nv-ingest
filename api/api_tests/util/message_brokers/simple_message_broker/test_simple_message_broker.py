# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import threading
import socket
import json
import time
from uuid import uuid4

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker

# Assuming the SimpleMessageBroker and related classes are imported from the module
# from your_module import SimpleMessageBroker

HOST = "127.0.0.1"
MAX_QUEUE_SIZE = 10
# Module-level current broker address, set by fixture for convenience
CURRENT_BROKER_ADDR = None


@pytest.fixture(scope="session")
def broker():
    """Start broker on an ephemeral port and yield (server, host, port)."""
    global CURRENT_BROKER_ADDR
    server = SimpleMessageBroker(HOST, 0, MAX_QUEUE_SIZE)
    host, port = server.server_address
    # Expose current address for helpers that don't get broker_addr explicitly
    CURRENT_BROKER_ADDR = (host, port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.2)
    try:
        yield server, host, port
    finally:
        # Best-effort shutdown to avoid hangs
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        # Avoid indefinite join in case shutdown is misbehaving
        try:
            server_thread.join(timeout=2.0)
        except Exception:
            pass
        # Clear module-level address and any singleton instance to prevent reuse
        try:
            CURRENT_BROKER_ADDR = None
        except Exception:
            pass
        try:
            from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker as _SMB

            setattr(_SMB, "_instance", None)
        except Exception:
            pass


@pytest.fixture
def broker_addr(broker):
    """Return (host, port) for the running broker."""
    _, host, port = broker
    return host, port


def send_request(request_data, broker_addr=None):
    """Helper method to send a request to the server and receive a response."""
    global CURRENT_BROKER_ADDR
    if broker_addr is None:
        if CURRENT_BROKER_ADDR is None:
            raise TypeError("send_request() missing required broker_addr and CURRENT_BROKER_ADDR is not set")
        host, port = CURRENT_BROKER_ADDR
    else:
        host, port = broker_addr
    sock = socket.create_connection((host, port))
    try:
        # Send request
        request_json = json.dumps(request_data).encode("utf-8")
        sock.sendall(len(request_json).to_bytes(8, "big") + request_json)

        # Receive initial response
        response_length_bytes = recv_exact(sock, 8)
        if not response_length_bytes:
            raise ConnectionError("Failed to receive response length.")
        response_length = int.from_bytes(response_length_bytes, "big")
        response_data_bytes = recv_exact(sock, response_length)
        if not response_data_bytes:
            raise ConnectionError("Failed to receive response data.")
        response_data = json.loads(response_data_bytes.decode("utf-8"))

        return sock, response_data
    except Exception as e:
        sock.close()
        raise e


def recv_exact(sock, num_bytes):
    """Helper method to receive an exact number of bytes."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def send_ack(sock, transaction_id, ack=True):
    """Helper method to send an ACK to the server."""
    ack_data = json.dumps({"transaction_id": transaction_id, "ack": ack}).encode("utf-8")
    sock.sendall(len(ack_data).to_bytes(8, "big") + ack_data)


def _push_message(queue_name, message, broker_addr=None):
    """Helper method to push a message into the queue without testing ACK behavior."""
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data, broker_addr)
    transaction_id = response["transaction_id"]
    send_ack(sock, transaction_id)
    # Receive final response
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    recv_exact(sock, response_length)  # Discard final response
    sock.close()


def test_push_with_ack(broker_addr):
    """Test PUSH operation with acknowledgment."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data, broker_addr)

    # Ensure initial response contains transaction ID
    assert response["response_code"] == 0
    assert "transaction_id" in response
    transaction_id = response["transaction_id"]

    # Send ACK
    send_ack(sock, transaction_id)

    # Receive final response
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 0
    assert final_response["response"] == "Data stored."
    sock.close()

    # Verify that the message is in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


def test_push_without_ack(broker_addr):
    """Test PUSH operation without acknowledgment (should not store the message)."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request
    request_data = {
        "command": "PUSH",
        "queue_name": queue_name,
        "message": message,
        "timeout": 1,  # Short timeout for the test
    }
    sock, response = send_request(request_data, broker_addr)

    # Do not send ACK, wait for timeout
    time.sleep(2)

    # Receive final response indicating failure
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 1
    assert final_response["response_reason"] == "ACK not received."
    sock.close()

    # Verify that the message is not in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


def test_pop_with_ack(broker_addr):
    """Test POP operation with acknowledgment."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Pre-populate the queue
    _push_message(queue_name, message, broker_addr)

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
    sock, response = send_request(request_data, broker_addr)

    # Ensure initial response contains the message and transaction ID
    assert response["response_code"] == 0
    assert "transaction_id" in response
    assert response["response"] == message
    transaction_id = response["transaction_id"]

    # Send ACK
    send_ack(sock, transaction_id)

    # Receive final response
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 0
    assert final_response["response"] == "Data processed."
    sock.close()

    # Verify that the queue is now empty
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


def test_pop_without_ack(broker_addr):
    """Test POP operation without acknowledgment (should return data to queue)."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Pre-populate the queue
    _push_message(queue_name, message, broker_addr)

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 1}  # Short timeout for the test
    sock, response = send_request(request_data, broker_addr)

    # Do not send ACK, wait for timeout
    time.sleep(2)

    # Receive final response indicating failure
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 1
    assert final_response["response_reason"] == "ACK not received."
    sock.close()

    # Verify that the message is back in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


def test_size_command(broker_addr):
    """Test SIZE command to get the number of items in a queue."""
    queue_name = f"test_queue_{uuid4()}"
    messages = ["Message 1", "Message 2", "Message 3"]

    # Push messages into the queue
    for msg in messages:
        _push_message(queue_name, msg, broker_addr)

    # Send SIZE request
    request_data = {"command": "SIZE", "queue_name": queue_name}
    sock, response = send_request(request_data, broker_addr)

    assert response["response_code"] == 0
    assert response["response"] == str(len(messages))
    sock.close()


def test_invalid_command(broker_addr):
    """Test handling of an invalid command."""
    request_data = {"command": "INVALID", "queue_name": "test_queue"}
    sock, response = send_request(request_data, broker_addr)

    assert response["response_code"] == 1
    assert response["response_reason"] == "Unknown command"
    sock.close()


def test_queue_not_exist(broker_addr):
    """Test POP from a non-existent queue."""
    queue_name = f"non_existent_queue_{uuid4()}"

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 1}
    sock, response = send_request(request_data)

    assert response["response_code"] == 2
    assert response["response_reason"] == "Job not ready"
    sock.close()


def test_queue_full(broker_addr):
    """Test PUSH operation when the queue is full."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Fill the queue to its maximum size
    for _ in range(MAX_QUEUE_SIZE):
        request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
        sock, response = send_request(request_data, broker_addr)

        # Receive initial response with transaction ID
        assert response["response_code"] == 0
        transaction_id = response["transaction_id"]

        # Send ACK
        send_ack(sock, transaction_id)

        # Receive final response indicating success
        response_length_bytes = recv_exact(sock, 8)
        response_length = int.from_bytes(response_length_bytes, "big")
        response_data_bytes = recv_exact(sock, response_length)
        final_response = json.loads(response_data_bytes.decode("utf-8"))

        assert final_response["response_code"] == 0
        assert final_response["response"] == "Data stored."
        sock.close()

    # Attempt to push another message beyond capacity
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data, broker_addr)

    # Receive immediate failure response (no transaction ID)
    assert response["response_code"] == 1
    assert response["response_reason"] == "Queue is full"

    # No ACK is required; close the socket
    sock.close()


def test_ack_with_wrong_transaction_id(broker_addr):
    """Test sending ACK with incorrect transaction ID."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data)

    # Send ACK with wrong transaction ID
    wrong_transaction_id = str(uuid4())
    send_ack(sock, wrong_transaction_id)

    # Receive final response indicating failure
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 1
    assert final_response["response_reason"] == "ACK not received."
    sock.close()

    # Verify that the message is not in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


def test_push_with_large_message(broker_addr):
    """Test pushing a large message to the queue."""
    queue_name = f"test_queue_{uuid4()}"
    message = "A" * (1024 * 1024 * 5)  # 5MB message

    # Send PUSH request
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data)
    transaction_id = response["transaction_id"]

    # Send ACK
    send_ack(sock, transaction_id)

    # Receive final response
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    response_data_bytes = recv_exact(sock, response_length)
    final_response = json.loads(response_data_bytes.decode("utf-8"))

    assert final_response["response_code"] == 0
    assert final_response["response"] == "Data stored."
    sock.close()

    # Verify that the message is in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request, broker_addr)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


def test_pop_from_empty_queue(broker_addr):
    """Test POP operation from an empty queue."""
    queue_name = f"test_queue_{uuid4()}"

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
    sock, response = send_request(request_data)

    assert response["response_code"] == 2
    assert response["response_reason"] == "Job not ready"
    sock.close()


def test_ack_timeout(broker_addr):
    """Test that the server times out waiting for ACK."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request with a short timeout
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 1}  # 1-second timeout
    sock, response = send_request(request_data, broker_addr)
    transaction_id = response["transaction_id"]

    # Wait for longer than the timeout
    time.sleep(2)

    # Attempt to send ACK (should be too late)
    send_ack(sock, transaction_id)

    # Receive final response indicating failure
    response_length_bytes = recv_exact(sock, 8)
    if response_length_bytes:
        response_length = int.from_bytes(response_length_bytes, "big")
        response_data_bytes = recv_exact(sock, response_length)
        final_response = json.loads(response_data_bytes.decode("utf-8"))

        # The server should have already timed out and discarded the data
        assert final_response["response_code"] == 1
        assert final_response["response_reason"] == "ACK not received."
    else:
        # Connection may have been closed by the server due to timeout
        pass
    sock.close()

    # Verify that the message is not in the queue
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


def test_concurrent_push_pop(broker_addr):
    """Test concurrent PUSH and POP operations."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(10)]
    results = []
    lock = threading.Lock()

    def push_message(msg):
        try:
            _push_message(queue_name, msg, broker_addr)
            with lock:
                results.append(f"Pushed: {msg}")
        except Exception as e:
            with lock:
                results.append(f"Push error: {e}")

    def pop_message():
        try:
            # Send POP request
            request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
            sock, response = send_request(request_data, broker_addr)
            if response["response_code"] == 0:
                transaction_id = response["transaction_id"]
                message = response["response"]
                # Send ACK
                send_ack(sock, transaction_id)
                # Receive final response
                response_length_bytes = recv_exact(sock, 8)
                response_length = int.from_bytes(response_length_bytes, "big")
                recv_exact(sock, response_length)  # Discard final response
                with lock:
                    results.append(f"Popped: {message}")
            else:
                with lock:
                    results.append(f"Pop error: {response['response_reason']}")
            sock.close()
        except Exception as e:
            with lock:
                results.append(f"Pop exception: {e}")

    # Start threads for pushing and popping
    threads = []
    for msg in messages:
        t_push = threading.Thread(target=push_message, args=(msg,))
        t_push.start()
        t_push.join()  # Wait for push to complete
        t_pop = threading.Thread(target=pop_message)
        t_pop.start()
        threads.append(t_pop)

    for t in threads:
        t.join()

    # Verify that all messages were pushed and popped
    pushed = [res for res in results if res.startswith("Pushed")]
    popped = [res for res in results if res.startswith("Popped")]
    push_errors = [res for res in results if res.startswith("Push error")]
    pop_errors = [res for res in results if res.startswith("Pop error") or res.startswith("Pop exception")]

    # Log any errors
    if push_errors or pop_errors:
        print("Errors encountered during test_concurrent_push_pop:")
        for error in push_errors + pop_errors:
            print(error)

    assert len(pushed) == len(messages)
    assert len(popped) == len(messages)


def test_multiple_clients(broker_addr):
    """Test multiple clients interacting with the server simultaneously."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Client {i} Message" for i in range(5)]
    lock = threading.Lock()
    results = []

    def client_thread(message):
        # Push message
        _push_message(queue_name, message, broker_addr)
        # Pop message
        request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
        sock, response = send_request(request_data, broker_addr)
        if response["response_code"] == 0:
            transaction_id = response["transaction_id"]
            popped_message = response["response"]
            # Send ACK
            send_ack(sock, transaction_id)
            # Receive final response
            response_length_bytes = recv_exact(sock, 8)
            response_length = int.from_bytes(response_length_bytes, "big")
            recv_exact(sock, response_length)  # Discard final response
            with lock:
                results.append(popped_message)
        sock.close()

    threads = [threading.Thread(target=client_thread, args=(msg,)) for msg in messages]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify that the queue is empty
    size_request = {"command": "SIZE", "queue_name": queue_name}
    sock, size_response = send_request(size_request)
    assert size_response["response"] == "0"
    sock.close()

    # Verify that all messages were processed
    assert sorted(results) == sorted(messages)
