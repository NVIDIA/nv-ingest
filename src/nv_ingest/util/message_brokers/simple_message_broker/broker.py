# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import queue
import socketserver
import json
import logging
from typing import Union, Optional
from pydantic import BaseModel, Field, ValidationError, Extra

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


# Define a ResponseSchema for all responses
class ResponseSchema(BaseModel):
    response_code: int
    response_reason: Optional[str] = "OK"
    response: Union[str, None] = ""  # Contains a message or data, or None if no data is returned


class SimpleMessageBrokerHandler(socketserver.BaseRequestHandler):
    """
    Request handler for SimpleMessageBroker using socketserver.
    Each client request is handled by this class.
    """

    def handle(self):
        client_address = self.client_address
        logger.debug(f"Handling client connection from {client_address}")

        try:
            # Receive the length of the incoming data
            data_length = int.from_bytes(self.request.recv(8), 'big')
            logger.debug(f"Expected data length: {data_length} bytes")

            # Read the data in chunks
            data = bytearray()
            while len(data) < data_length:
                packet = self.request.recv(min(4096, data_length - len(data)))
                if not packet:
                    break
                data.extend(packet)
            data = data.decode('utf-8').strip()
            logger.debug(f"Received data from {client_address}: {data[:100]}...")  # Log first 100 chars for brevity

            # If no data is received, assume client disconnected
            if not data:
                logger.debug(f"No data received. Closing connection to {client_address}")
                return

            # Parse JSON data for the command
            request_data = json.loads(data)
            command = request_data.get("command")
            queue_name = request_data.get("queue_name")
            message = request_data.get("message", "")
            timeout = request_data.get("timeout", None)

            logger.debug(f"Parsed command: {command} with queue_name: {queue_name}")

            # Process the command and get a structured response
            if command == "PUSH":
                response_schema = self.server.broker._push_to_queue(queue_name, message, timeout)
            elif command == "POP":
                response_schema = self.server.broker._pop_from_queue(queue_name, timeout)
            else:
                response_schema = ResponseSchema(response_code=1, response_reason="Unknown command")

            # Send the structured JSON response back to the client in chunks
            response_json = response_schema.json().encode('utf-8')
            self._send_large_message(response_json)

            logger.debug(f"Sent response to {client_address}: {response_json[:100]}...")  # Log first 100 chars

        except Exception as e:
            logger.error(f"Error processing command from {client_address}: {e}")

    def _send_large_message(self, message: bytes):
        """Send large messages in chunks with length header."""
        self.request.sendall(len(message).to_bytes(8, 'big'))  # Send the length of the message first
        for i in range(0, len(message), 4096):
            self.request.sendall(message[i:i + 4096])


class SimpleMessageBroker(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    A simple message broker that accepts PUSH and POP commands over TCP,
    using socketserver for connection handling.
    """
    allow_reuse_address = True

    def __init__(self, host: str, port: int, max_queue_size: int):
        super().__init__((host, port), SimpleMessageBrokerHandler)
        self.max_queue_size = max_queue_size
        self.queues = {}

    def _process_command(self, command: str, args: list) -> ResponseSchema:
        try:
            if command == "PUSH":
                push_request = PushRequestSchema(queue_name=args[0], message=" ".join(args[1:]))
                return self._push_to_queue(push_request.queue_name, push_request.message, push_request.timeout)
            elif command == "POP":
                pop_request = PopRequestSchema(queue_name=args[0])
                return self._pop_from_queue(pop_request.queue_name, pop_request.timeout)
            else:
                return ResponseSchema(response_code=1, response_reason=f"Unknown verb: {command}")
        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            return ResponseSchema(response_code=1, response_reason=f"Validation Error: {ve}")

    def _push_to_queue(self, queue_name: str, message: str, timeout: Optional[float] = None) -> ResponseSchema:
        if queue_name not in self.queues:
            self.queues[queue_name] = queue.Queue(maxsize=self.max_queue_size)

        try:
            self.queues[queue_name].put(message, block=True, timeout=timeout)  # Blocking put with timeout
            return ResponseSchema(response_code=0, response="OK")
        except queue.Full:
            logger.debug(f"Queue '{queue_name}' is full. Cannot add message: {message}")
            return ResponseSchema(response_code=1, response_reason="Queue is full")

    def _pop_from_queue(self, queue_name: str, timeout: Optional[float] = None) -> ResponseSchema:
        if queue_name not in self.queues:
            return ResponseSchema(response_code=1, response_reason="Queue does not exist")

        try:
            message = self.queues[queue_name].get(block=True, timeout=timeout)  # Blocking get with timeout
            return ResponseSchema(response_code=0, response=message)
        except queue.Empty:
            return ResponseSchema(response_code=1, response_reason="Queue is empty")
