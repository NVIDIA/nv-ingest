# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
import queue
import socket
import socketserver
import json
import logging
import threading
from typing import Union, Optional

from pydantic import BaseModel, Field, ValidationError, Extra

from nv_ingest.util.message_brokers.simple_message_broker.ordered_message_queue import OrderedMessageQueue

logger = logging.getLogger(__name__)


# Define schemas for request validation
class PushRequestSchema(BaseModel):
    queue_name: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    timeout: Optional[float] = None  # Optional timeout for blocking push

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments


class PopRequestSchema(BaseModel):
    queue_name: str = Field(..., min_length=1)
    timeout: Optional[float] = None  # Optional timeout for blocking pop

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments


class SizeRequestSchema(BaseModel):
    queue_name: str = Field(..., min_length=1)

    class Config:
        extra = Extra.forbid  # Prevents any extra arguments


class ResponseSchema(BaseModel):
    response_code: int
    response_reason: Optional[str] = "OK"
    response: Union[str, dict, None] = None
    transaction_id: Optional[str] = None  # Unique transaction ID


class SimpleMessageBrokerHandler(socketserver.BaseRequestHandler):
    def handle(self):
        client_address = self.client_address
        logger.debug(f"Handling client connection from {client_address}")

        try:
            data_length_bytes = self._recv_exact(8)
            if not data_length_bytes:
                logger.debug("No data length received. Closing connection.")
                return
            data_length = int.from_bytes(data_length_bytes, 'big')

            data_bytes = self._recv_exact(data_length)
            if not data_bytes:
                logger.debug("No data received. Closing connection.")
                return
            data = data_bytes.decode('utf-8').strip()
            request_data = json.loads(data)
            logger.debug(f"Received data: {request_data}")

            command = request_data.get("command")
            queue_name = request_data.get("queue_name")
            message = request_data.get("message", "")
            timeout = request_data.get("timeout", None)

            transaction_id = str(uuid.uuid4())

            # Ensure the queue and its lock exist
            self.server._initialize_queue(queue_name)
            queue_lock = self.server.queue_locks[queue_name]
            queue = self.server.queues[queue_name]

            if command == "PUSH":
                self._handle_push(queue_name, message, timeout, transaction_id, queue, queue_lock)
            elif command == "POP":
                self._handle_pop(queue_name, timeout, transaction_id, queue, queue_lock)
            elif command == "SIZE":
                response = self._size_of_queue(queue_name, queue, queue_lock)
                self._send_response(response)
            else:
                response = ResponseSchema(response_code=1, response_reason="Unknown command")
                self._send_response(response)

        except Exception as e:
            logger.error(f"Error processing command from {client_address}: {e}")
            response = ResponseSchema(response_code=1, response_reason=str(e))
            self._send_response(response)

    def _handle_push(self, queue_name: str, message: str, timeout: Optional[float],
                     transaction_id: str, queue: OrderedMessageQueue, queue_lock: threading.Lock):
        with queue_lock:
            if queue.full():
                # Return failure response immediately
                response = ResponseSchema(response_code=1, response_reason="Queue is full")
                self._send_response(response)
                return

        # Proceed with the 3-way handshake
        # Send initial response with transaction ID
        initial_response = ResponseSchema(
            response_code=0,
            response="Transaction initiated. Waiting for ACK.",
            transaction_id=transaction_id
        )
        self._send_response(initial_response)

        # Wait for ACK
        if not self._wait_for_ack(transaction_id, timeout):
            logger.debug(f"Transaction {transaction_id}: ACK not received. Discarding data.")
            final_response = ResponseSchema(response_code=1, response_reason="ACK not received.",
                                            transaction_id=transaction_id)
        else:
            # Perform the PUSH operation after ACK
            with queue_lock:
                queue.push(message)
            final_response = ResponseSchema(response_code=0, response="Data stored.", transaction_id=transaction_id)

        # Send final response
        self._send_response(final_response)

    def _handle_pop(self, queue_name: str, timeout: Optional[float],
                    transaction_id: str, queue: OrderedMessageQueue, queue_lock: threading.Lock):
        with queue_lock:
            if queue.empty():
                # Return failure response immediately
                response = ResponseSchema(response_code=1, response_reason="Queue is empty")
                self._send_response(response)
                return
            # Pop the message from the queue
            message = queue.pop(transaction_id)

        # Proceed with the 3-way handshake
        # Send initial response with transaction ID and message
        initial_response = ResponseSchema(
            response_code=0,
            response=message,
            transaction_id=transaction_id
        )
        self._send_response(initial_response)

        # Wait for ACK
        if not self._wait_for_ack(transaction_id, timeout):
            logger.debug(f"Transaction {transaction_id}: ACK not received. Returning data to queue.")
            with queue_lock:
                queue.return_message(transaction_id)
            final_response = ResponseSchema(response_code=1, response_reason="ACK not received.",
                                            transaction_id=transaction_id)
        else:
            with queue_lock:
                queue.acknowledge(transaction_id)
            final_response = ResponseSchema(response_code=0, response="Data processed.", transaction_id=transaction_id)

        # Send final response
        self._send_response(final_response)

    def _size_of_queue(self, queue_name: str, queue: OrderedMessageQueue, queue_lock: threading.Lock) -> ResponseSchema:
        with queue_lock:
            size = queue.qsize()
        return ResponseSchema(response_code=0, response=str(size))

    def _wait_for_ack(self, transaction_id: str, timeout: Optional[float]) -> bool:
        try:
            self.request.settimeout(timeout)
            ack_length_bytes = self._recv_exact(8)
            if not ack_length_bytes or len(ack_length_bytes) < 8:
                return False
            ack_length = int.from_bytes(ack_length_bytes, 'big')
            ack_data_bytes = self._recv_exact(ack_length)
            if not ack_data_bytes or len(ack_data_bytes) < ack_length:
                return False
            ack_data = ack_data_bytes.decode('utf-8')
            ack_response = json.loads(ack_data)
            return ack_response.get("transaction_id") == transaction_id and ack_response.get("ack") is True
        except (socket.timeout, json.JSONDecodeError, ConnectionResetError) as e:
            logger.error(f"Error waiting for ACK: {e}")
            return False
        finally:
            self.request.settimeout(None)

    def _send_response(self, response: ResponseSchema):
        response_json = response.json().encode('utf-8')
        total_length = len(response_json)
        self.request.sendall(total_length.to_bytes(8, 'big'))
        self.request.sendall(response_json)

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """Helper method to receive an exact number of bytes."""
        data = bytearray()
        while len(data) < num_bytes:
            packet = self.request.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)


class SimpleMessageBroker(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int, max_queue_size: int):
        super().__init__((host, port), SimpleMessageBrokerHandler)
        self.max_queue_size = max_queue_size
        self.queues = {}
        self.queue_locks = {}  # Dictionary to hold locks for each queue
        self.lock = threading.Lock()  # Global lock to protect access to queues and locks

    def _initialize_queue(self, queue_name: str):
        with self.lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = OrderedMessageQueue(maxsize=self.max_queue_size)
                self.queue_locks[queue_name] = threading.Lock()

    def _push_to_queue(self, queue_name: str, message: str) -> ResponseSchema:
        try:
            self.queues[queue_name].put_nowait(message)
            return ResponseSchema(response_code=0, response="Data stored.")
        except queue.Full:
            logger.debug(f"Queue '{queue_name}' is full. Cannot add message.")
            return ResponseSchema(response_code=1, response_reason="Queue is full")

    def _pop_from_queue(self, queue_name: str) -> ResponseSchema:
        try:
            message = self.queues[queue_name].get_nowait()
            return ResponseSchema(response_code=0, response=message)
        except queue.Empty:
            return ResponseSchema(response_code=1, response_reason="Queue is empty")