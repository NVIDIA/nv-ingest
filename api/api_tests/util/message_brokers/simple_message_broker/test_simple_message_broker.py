# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

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
PORT = 2000 + random.randint(0, 10000)  # Use an available port
MAX_QUEUE_SIZE = 5


@pytest.fixture(scope="module")
def broker_server():
    """Fixture to start and stop the SimpleMessageBroker server."""
    server = SimpleMessageBroker(HOST, PORT, MAX_QUEUE_SIZE)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Allows program to exit even if thread is running
    server_thread.start()
    time.sleep(1)  # Give the server a moment to start
    yield

    server.shutdown()
    server.server_close()
    server_thread.join()


def send_request(request_data):
    """Helper method to send a request to the server and receive a response."""
    sock = socket.create_connection((HOST, PORT))
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


def _push_message(queue_name, message):
    """Helper method to push a message into the queue without testing ACK behavior."""
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data)
    transaction_id = response["transaction_id"]
    send_ack(sock, transaction_id)
    # Receive final response
    response_length_bytes = recv_exact(sock, 8)
    response_length = int.from_bytes(response_length_bytes, "big")
    recv_exact(sock, response_length)  # Discard final response
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_push_with_ack():
    """Test PUSH operation with acknowledgment."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
    sock, response = send_request(request_data)

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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_push_without_ack():
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
    sock, response = send_request(request_data)

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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_pop_with_ack():
    """Test POP operation with acknowledgment."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Pre-populate the queue
    _push_message(queue_name, message)

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
    sock, response = send_request(request_data)

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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_pop_without_ack():
    """Test POP operation without acknowledgment (should return data to queue)."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Pre-populate the queue
    _push_message(queue_name, message)

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 1}  # Short timeout for the test
    sock, response = send_request(request_data)

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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_size_command():
    """Test SIZE command to get the number of items in a queue."""
    queue_name = f"test_queue_{uuid4()}"
    messages = ["Message 1", "Message 2", "Message 3"]

    # Push messages into the queue
    for msg in messages:
        _push_message(queue_name, msg)

    # Send SIZE request
    request_data = {"command": "SIZE", "queue_name": queue_name}
    sock, response = send_request(request_data)

    assert response["response_code"] == 0
    assert response["response"] == str(len(messages))
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_invalid_command():
    """Test handling of an invalid command."""
    request_data = {"command": "INVALID", "queue_name": "test_queue"}
    sock, response = send_request(request_data)

    assert response["response_code"] == 1
    assert response["response_reason"] == "Unknown command"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_queue_not_exist():
    """Test POP from a non-existent queue."""
    queue_name = f"non_existent_queue_{uuid4()}"

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 1}
    sock, response = send_request(request_data)

    assert response["response_code"] == 2
    assert response["response_reason"] == "Job not ready"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_queue_full():
    """Test PUSH operation when the queue is full."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Fill the queue to its maximum size
    for _ in range(MAX_QUEUE_SIZE):
        request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 5}
        sock, response = send_request(request_data)

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
    sock, response = send_request(request_data)

    # Receive immediate failure response (no transaction ID)
    assert response["response_code"] == 1
    assert response["response_reason"] == "Queue is full"

    # No ACK is required; close the socket
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_ack_with_wrong_transaction_id():
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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "0"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_push_with_large_message():
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
    sock, size_response = send_request(size_request)
    assert size_response["response_code"] == 0
    assert size_response["response"] == "1"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_pop_from_empty_queue():
    """Test POP operation from an empty queue."""
    queue_name = f"test_queue_{uuid4()}"

    # Send POP request
    request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
    sock, response = send_request(request_data)

    assert response["response_code"] == 2
    assert response["response_reason"] == "Job not ready"
    sock.close()


@pytest.mark.usefixtures("broker_server")
def test_ack_timeout():
    """Test that the server times out waiting for ACK."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Send PUSH request with a short timeout
    request_data = {"command": "PUSH", "queue_name": queue_name, "message": message, "timeout": 1}  # 1-second timeout
    sock, response = send_request(request_data)
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


def test_concurrent_push_pop():
    """Test concurrent PUSH and POP operations."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(10)]
    results = []
    lock = threading.Lock()

    def push_message(msg):
        try:
            _push_message(queue_name, msg)
            with lock:
                results.append(f"Pushed: {msg}")
        except Exception as e:
            with lock:
                results.append(f"Push error: {e}")

    def pop_message():
        try:
            # Send POP request
            request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
            sock, response = send_request(request_data)
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


@pytest.mark.usefixtures("broker_server")
def test_multiple_clients():
    """Test multiple clients interacting with the server simultaneously."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Client {i} Message" for i in range(5)]
    lock = threading.Lock()
    results = []

    def client_thread(message):
        # Push message
        _push_message(queue_name, message)
        # Pop message
        request_data = {"command": "POP", "queue_name": queue_name, "timeout": 5}
        sock, response = send_request(request_data)
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
