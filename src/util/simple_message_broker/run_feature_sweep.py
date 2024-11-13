# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import socket
import threading
import time
from typing import Dict

from nv_ingest.util.message_brokers.simple_message_broker import SimpleMessageBroker
from nv_ingest.util.message_brokers.simple_message_broker.broker import ResponseSchema

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Error collection list to log all errors at the end of the sweep
error_log = []


def truncate_response(response_schema: Dict):
    truncated_response = response_schema.get("response", '')
    response_code = response_schema.get("response_code", 0)
    response_reason = response_schema.get("response_reason", "OK")

    if isinstance(truncated_response, str) and len(truncated_response) > 80:
        truncated_response = truncated_response[:80] + "..."

    response_str = f"ResponseSchema(response_code={response_code}, " \
                   f"response_reason='{response_reason}', " \
                   f"response='{truncated_response}')"  # Truncate long responses

    return response_str


def start_broker_server(host='127.0.0.1', port=63290, max_queue_size=100):
    """Start the SimpleMessageBroker server."""
    broker = SimpleMessageBroker(host, port, max_queue_size)
    broker.broker = broker  # Attach the broker instance to the server to access _process_command
    logger.info(f"SimpleMessageBroker server started on {host}:{port}")

    # Start the server in a separate thread to allow it to run in the background
    server_thread = threading.Thread(target=broker.serve_forever, daemon=True)
    server_thread.start()
    return broker, server_thread


def stop_broker_server(broker: SimpleMessageBroker):
    """Stops the SimpleMessageBroker server."""
    logger.info("Stopping SimpleMessageBroker server...")
    broker.shutdown()
    broker.server_close()
    logger.info("SimpleMessageBroker server stopped.")


# Start SimpleMessageBroker using start_broker_server function
broker_instance, broker_thread = start_broker_server(host='127.0.0.1', port=63790, max_queue_size=5)
logger.debug("SimpleMessageBroker server started.")

# Give the broker some time to start
time.sleep(1)


# Helper to send large messages in chunks
def send_large_message(sock, message):
    total_length = len(message)
    sock.sendall(total_length.to_bytes(8, 'big'))  # Send message length first
    for i in range(0, total_length, 4096):  # Send in 4KB chunks
        sock.sendall(message[i:i + 4096])


# Helper to receive large messages in chunks
def receive_large_message(sock):
    total_length = int.from_bytes(sock.recv(8), 'big')  # Receive message length first
    chunks = []
    bytes_recd = 0
    while bytes_recd < total_length:
        chunk = sock.recv(min(total_length - bytes_recd, 4096))
        if not chunk:
            break
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b''.join(chunks).decode('utf-8')


# Test client function to push messages to a queue
def push_message(queue_name, message, timeout=None):
    error_response = ResponseSchema(response_code=1, response="")
    try:
        with socket.create_connection(("127.0.0.1", 63790)) as sock:
            # Include timeout in the command payload if provided
            command = {"command": "PUSH", "queue_name": queue_name, "message": message}
            if timeout is not None:
                command["timeout"] = timeout
            command_json = json.dumps(command)

            logger.debug(f"Sending PUSH command for message of size {len(message)} bytes with timeout {timeout}.")
            send_large_message(sock, command_json.encode('utf-8'))
            response_data = receive_large_message(sock)

            # Parse JSON response
            response = json.loads(response_data)
            logger.debug(f"Pushed message of size {len(message)} | Response: {truncate_response(response)}")

            return response
    except ConnectionRefusedError:
        error_log.append("Connection refused. Ensure the SimpleMessageBroker is running.")
        error_response.response_reason = "Connection refused"
        return error_response.dict()
    except Exception as e:
        error_log.append(f"Error in push_message: {e}")
        error_response.response_reason = str(e)
        return error_response.dict()


# Test client function to pop messages from a queue with an optional timeout
def pop_message(queue_name, timeout=None):
    error_response = ResponseSchema(response_code=1, response="")
    try:
        with socket.create_connection(("127.0.0.1", 63790), timeout=5) as sock:
            # Include timeout in the command payload if provided
            command = {"command": "POP", "queue_name": queue_name}
            if timeout is not None:
                command["timeout"] = timeout
            command_json = json.dumps(command)

            logger.debug(f"Sending POP command with timeout {timeout}.")
            send_large_message(sock, command_json.encode('utf-8'))
            response_data = receive_large_message(sock)

            # Parse JSON response
            response = json.loads(response_data)
            logger.debug(f"Popped from {queue_name} | Response: {truncate_response(response)}")

            return response  # Return the message or empty if None
    except (ConnectionRefusedError, socket.timeout) as e:
        error_log.append(f"Connection error or timeout: {e}")
        error_response.response_reason = str(e)
        return error_response.dict()
    except Exception as e:
        error_log.append(f"Error in pop_message: {e}")
        error_response.response_reason = str(e)
        return error_response.dict()


try:
    # Test pushing small messages to the queue
    logger.debug("Starting push tests")
    push_message("test_queue", "Hello, world!")
    push_message("test_queue", "Another message")
    push_message("test_queue", "Message 3")

    # Test popping messages from the queue
    logger.debug("Starting pop tests")
    pop_message("test_queue")
    pop_message("test_queue")
    pop_message("test_queue")

    # Generate and push a large 150MB message
    large_message = "A" * (150 * 1024 * 1024)  # 150MB message
    logger.debug(f"Testing with large message of size {len(large_message)} bytes.")
    push_response = push_message("test_queue", large_message)
    if push_response.get("response_code") != 0:
        error_log.append(f"Failed to push 150MB message: {push_response}")

    # Pop the large message from the queue and verify its size
    pop_response = pop_message("test_queue")
    if pop_response.get("response_code") != 0 or len(pop_response.get("response", "")) != len(large_message):
        error_log.append(f"Failed to correctly pop 150MB message: {pop_response}")

    # Test blocking push with timeout (expecting timeout error)
    logger.debug("Testing blocking push with timeout")
    for i in range(5):
        push_message("test_queue", f"Message {i}")

    # Queue is now full, this should timeout
    timeout_push_response = push_message("test_queue", "Overflow message", timeout=2)
    if timeout_push_response.get("response_code") == 0:
        error_log.append("Expected timeout error for blocking push, but got success.")
    elif timeout_push_response.get("response_reason") != "Queue is full":
        error_log.append(f"Unexpected response for timeout push: {timeout_push_response}")

    # Test blocking pop with timeout (expecting timeout error)
    logger.debug("Testing blocking pop with timeout")
    for _ in range(5):  # Empty the queue
        pop_message("test_queue")

    # Queue is now empty, this should timeout
    timeout_pop_response = pop_message("test_queue", timeout=2)
    if timeout_pop_response.get("response_code") == 0:
        error_log.append("Expected timeout error for blocking pop, but got success.")
    elif timeout_pop_response.get("response_reason") != "Queue is empty":
        error_log.append(f"Unexpected response for timeout pop: {timeout_pop_response}")

    logger.debug("Test script completed.")

finally:
    # Ensure the broker is stopped after tests
    stop_broker_server(broker_instance)
    broker_thread.join()  # Wait for the broker thread to finish

# Log all collected errors at the end of the sweep
if error_log:
    logger.error("Summary of API validation errors:")
    for error in error_log:
        logger.error(error)
else:
    logger.info("All API validation tests passed successfully.")
