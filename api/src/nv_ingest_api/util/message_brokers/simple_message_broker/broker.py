# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
import socket
import socketserver
import json
import logging
import threading
from typing import Optional

from pydantic import ValidationError

from nv_ingest_api.internal.schemas.message_brokers.request_schema import (
    PushRequestSchema,
    PopRequestSchema,
    SizeRequestSchema,
)
from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema
from nv_ingest_api.util.message_brokers.simple_message_broker.ordered_message_queue import OrderedMessageQueue

logger = logging.getLogger(__name__)


class SimpleMessageBrokerHandler(socketserver.BaseRequestHandler):
    """
    Handles incoming client requests for the SimpleMessageBroker server, processes commands such as
    PUSH, POP, SIZE, and PING, and manages message queues with thread-safe operations.
    """

    def handle(self):
        """
        Handles incoming client requests, validates the request data, and dispatches to the appropriate
        command handler.

        Raises
        ------
        ValidationError
            If the incoming request data fails schema validation.
        Exception
            If there is an unexpected error while processing the request.
        """

        client_address = self.client_address

        data_bytes = None
        try:
            data_length_bytes = self._recv_exact(8)
            if not data_length_bytes:
                logger.debug("No data length received. Closing connection.")
                return
            data_length = int.from_bytes(data_length_bytes, "big")

            data_bytes = self._recv_exact(data_length)
            if not data_bytes:
                logger.debug("No data received. Closing connection.")
                return

            data = data_bytes.decode("utf-8").strip()
            request_data = json.loads(data)

            command = request_data.get("command")
            if not command:
                response = ResponseSchema(response_code=1, response_reason="No command specified")
                self._send_response(response)
                return

            # Handle the PING command directly
            if command == "PING":
                self._handle_ping()
                return

            # Validate and extract common fields
            queue_name = request_data.get("queue_name")

            # Initialize the queue and its lock if necessary
            if queue_name:
                self.server._initialize_queue(queue_name)
                queue_lock = self.server.queue_locks[queue_name]
                queue = self.server.queues[queue_name]
            else:
                queue_lock = None
                queue = None  # For commands that don't require a queue

            # Dispatch to the appropriate handler
            if command == "PUSH":
                validated_data = PushRequestSchema(**request_data)
                self._handle_push(validated_data, transaction_id=str(uuid.uuid4()), queue=queue, queue_lock=queue_lock)
            elif command == "PUSH_FOR_NV_INGEST":
                validated_data = PushRequestSchema(**request_data)
                self._handle_push_for_nv_ingest(validated_data, queue=queue, queue_lock=queue_lock)
            elif command == "POP":
                validated_data = PopRequestSchema(**request_data)
                self._handle_pop(validated_data, queue=queue, queue_lock=queue_lock)
            elif command == "SIZE":
                validated_data = SizeRequestSchema(**request_data)
                response = self._size_of_queue(validated_data, queue, queue_lock)
                self._send_response(response)
            else:
                response = ResponseSchema(response_code=1, response_reason="Unknown command")
                self._send_response(response)

        except ValidationError as ve:
            response = ResponseSchema(response_code=1, response_reason=str(ve))
            self._send_response(response)
        except Exception as e:
            logger.error(f"Error processing command from {client_address}: {e}\n{data_bytes}")
            response = ResponseSchema(response_code=1, response_reason=str(e))
            try:
                self._send_response(response)
            except BrokenPipeError:
                logger.error("Cannot send error response; client connection closed.")

    def _handle_ping(self):
        """
        Responds to a PING command with a PONG response.
        """

        response = ResponseSchema(response_code=0, response="PONG")
        self._send_response(response)

    def _handle_push(
        self, data: PushRequestSchema, transaction_id: str, queue: OrderedMessageQueue, queue_lock: threading.Lock
    ):
        """
        Handles a PUSH command to add a message to the specified queue.

        Parameters
        ----------
        data : PushRequestSchema
            The validated data for the PUSH command.
        transaction_id : str
            The unique transaction ID for the operation.
        queue : OrderedMessageQueue
            The queue where the message will be pushed.
        queue_lock : threading.Lock
            The lock object to ensure thread-safe access to the queue.
        """

        timeout = data.timeout

        with queue_lock:
            if queue.full():
                # Return failure response immediately
                response = ResponseSchema(response_code=1, response_reason="Queue is full")
                self._send_response(response)
                return

        # Proceed with the 3-way handshake
        initial_response = ResponseSchema(
            response_code=0, response="Transaction initiated. Waiting for ACK.", transaction_id=transaction_id
        )
        self._send_response(initial_response)

        # Wait for ACK
        if not self._wait_for_ack(transaction_id, timeout):
            logger.debug(f"Transaction {transaction_id}: ACK not received. Discarding data.")
            final_response = ResponseSchema(
                response_code=1, response_reason="ACK not received.", transaction_id=transaction_id
            )
        else:
            # Perform the PUSH operation after ACK
            with queue_lock:
                queue.push(data.message)
            final_response = ResponseSchema(response_code=0, response="Data stored.", transaction_id=transaction_id)

        # Send final response
        self._send_response(final_response)

    def _handle_push_for_nv_ingest(
        self, data: PushRequestSchema, queue: OrderedMessageQueue, queue_lock: threading.Lock
    ):
        """
        Handles a PUSH_FOR_NV_INGEST command, which includes generating a unique job ID and
        updating the message payload accordingly.

        Parameters
        ----------
        data : PushRequestSchema
            The validated data for the PUSH_FOR_NV_INGEST command.
        queue : OrderedMessageQueue
            The queue where the message will be pushed.
        queue_lock : threading.Lock
            The lock object to ensure thread-safe access to the queue.
        """

        timeout = data.timeout

        # Deserialize the message
        try:
            message_dict = json.loads(data.message)
        except json.JSONDecodeError:
            response = ResponseSchema(response_code=1, response_reason="Invalid JSON message")
            self._send_response(response)
            return

        # Generate a UUID for 'job_id' and use it as transaction_id
        transaction_id = str(uuid.uuid4())
        message_dict["job_id"] = transaction_id

        # Re-serialize the message
        updated_message = json.dumps(message_dict)

        with queue_lock:
            if queue.full():
                # Return failure response immediately
                response = ResponseSchema(response_code=1, response_reason="Queue is full")
                self._send_response(response)
                return

        # Proceed with the 3-way handshake
        initial_response = ResponseSchema(
            response_code=0, response="Transaction initiated. Waiting for ACK.", transaction_id=transaction_id
        )
        self._send_response(initial_response)

        # Wait for ACK
        if not self._wait_for_ack(transaction_id, timeout):
            logger.debug(f"Transaction {transaction_id}: ACK not received. Discarding data.")
            final_response = ResponseSchema(
                response_code=1, response_reason="ACK not received.", transaction_id=transaction_id
            )
        else:
            # Perform the PUSH operation after ACK
            with queue_lock:
                queue.push(updated_message)
            final_response = ResponseSchema(response_code=0, response="Data stored.", transaction_id=transaction_id)

        # Send final response
        self._send_response(final_response)

    def _handle_pop(self, data: PopRequestSchema, queue: OrderedMessageQueue, queue_lock: threading.Lock):
        """
        Handles a POP command to retrieve a message from the specified queue.

        Parameters
        ----------
        data : PopRequestSchema
            The validated data for the POP command.
        queue : OrderedMessageQueue
            The queue from which the message will be retrieved.
        queue_lock : threading.Lock
            The lock object to ensure thread-safe access to the queue.
        """

        timeout = data.timeout
        transaction_id = str(uuid.uuid4())

        with queue_lock:
            if queue.empty():
                # Return failure response immediately
                response = ResponseSchema(response_code=2, response_reason="Job not ready")
                self._send_response(response)
                return
            # Pop the message from the queue
            message = queue.pop(transaction_id)

        # Proceed with the 3-way handshake
        initial_response = ResponseSchema(response_code=0, response=message, transaction_id=transaction_id)
        self._send_response(initial_response)

        # Wait for ACK
        if not self._wait_for_ack(transaction_id, timeout):
            logger.debug(f"Transaction {transaction_id}: ACK not received. Returning data to queue.")
            with queue_lock:
                queue.return_message(transaction_id)
            final_response = ResponseSchema(
                response_code=1, response_reason="ACK not received.", transaction_id=transaction_id
            )
        else:
            with queue_lock:
                queue.acknowledge(transaction_id)
            final_response = ResponseSchema(response_code=0, response="Data processed.", transaction_id=transaction_id)

        # Send final response
        self._send_response(final_response)

    def _size_of_queue(
        self, data: SizeRequestSchema, queue: OrderedMessageQueue, queue_lock: threading.Lock
    ) -> ResponseSchema:
        """
        Retrieves the size of the specified queue.

        Parameters
        ----------
        data : SizeRequestSchema
            The validated data for the SIZE command.
        queue : OrderedMessageQueue
            The queue whose size will be queried.
        queue_lock : threading.Lock
            The lock object to ensure thread-safe access to the queue.

        Returns
        -------
        ResponseSchema
            A response containing the size of the queue.
        """

        with queue_lock:
            size = queue.qsize()
        return ResponseSchema(response_code=0, response=str(size))

    def _wait_for_ack(self, transaction_id: str, timeout: Optional[float]) -> bool:
        """
        Waits for an acknowledgment (ACK) from the client for a specific transaction.

        Parameters
        ----------
        transaction_id : str
            The unique transaction ID for the operation.
        timeout : float, optional
            The timeout period for waiting for the ACK.

        Returns
        -------
        bool
            True if the ACK is received, False otherwise.
        """

        try:
            self.request.settimeout(timeout)
            ack_length_bytes = self._recv_exact(8)
            if not ack_length_bytes or len(ack_length_bytes) < 8:
                return False
            ack_length = int.from_bytes(ack_length_bytes, "big")
            ack_data_bytes = self._recv_exact(ack_length)
            if not ack_data_bytes or len(ack_data_bytes) < ack_length:
                return False
            ack_data = ack_data_bytes.decode("utf-8")
            ack_response = json.loads(ack_data)
            return ack_response.get("transaction_id") == transaction_id and ack_response.get("ack") is True
        except (socket.timeout, json.JSONDecodeError, ConnectionResetError) as e:
            logger.error(f"Error waiting for ACK: {e}")
            return False
        finally:
            self.request.settimeout(None)

    def _send_response(self, response: ResponseSchema):
        """
        Sends a response back to the client.

        Parameters
        ----------
        response : ResponseSchema
            The response to send to the client.

        Raises
        ------
        Exception
            If there is an error while sending the response.
        """

        try:
            response_json = response.model_dump_json().encode("utf-8")
            total_length = len(response_json)
            self.request.sendall(total_length.to_bytes(8, "big"))
            self.request.sendall(response_json)
        except BrokenPipeError as e:
            logger.error(f"BrokenPipeError while sending response: {e}")
            # Handle the broken pipe gracefully
        except Exception as e:
            logger.error(f"Unexpected error while sending response: {e}")

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """
        Receives an exact number of bytes from the client connection.

        Parameters
        ----------
        num_bytes : int
            The number of bytes to receive.

        Returns
        -------
        Optional[bytes]
            The received bytes, or None if the connection is closed.
        """

        data = bytearray()
        while len(data) < num_bytes:
            packet = self.request.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)


class SimpleMessageBroker(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    A thread-safe message broker server that manages multiple message queues and supports commands
    such as PUSH, POP, SIZE, and PING.
    """

    allow_reuse_address = True
    _instances = {}
    _instances_lock = threading.Lock()

    def __new__(cls, host: str, port: int, max_queue_size: int):
        """
        Ensures that only one instance of SimpleMessageBroker is created per host and port combination.

        Parameters
        ----------
        host : str
            The hostname or IP address for the server.
        port : int
            The port number for the server.
        max_queue_size : int
            The maximum size of each message queue.

        Returns
        -------
        SimpleMessageBroker
            The singleton instance of the server.
        """

        key = (host, port)
        with cls._instances_lock:
            if key not in cls._instances:
                # Create a new instance and store it in the instances dictionary
                instance = super(SimpleMessageBroker, cls).__new__(cls)
                cls._instances[key] = instance
            else:
                instance = cls._instances[key]
        return instance

    def __init__(self, host: str, port: int, max_queue_size: int):
        """
        Initializes the SimpleMessageBroker server, setting up message queues and locks.

        Parameters
        ----------
        host : str
            The hostname or IP address for the server.
        port : int
            The port number for the server.
        max_queue_size : int
            The maximum size of each message queue.
        """

        # Prevent __init__ from running multiple times on the same instance
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__((host, port), SimpleMessageBrokerHandler)
        self.max_queue_size = max_queue_size
        self.queues = {}
        self.queue_locks = {}  # Dictionary to hold locks for each queue
        self.lock = threading.Lock()  # Global lock to protect access to queues and locks
        self._initialized = True  # Flag to indicate initialization is complete

    def _initialize_queue(self, queue_name: str):
        """
        Initializes a new message queue with the specified name if it doesn't already exist.

        Parameters
        ----------
        queue_name : str
            The name of the queue to initialize.
        """

        with self.lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = OrderedMessageQueue(maxsize=self.max_queue_size)
                self.queue_locks[queue_name] = threading.Lock()
