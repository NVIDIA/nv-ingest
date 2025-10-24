# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This code is duplicated from the ingest service:
# src/nv_ingest/util/message_brokers/simple_message_broker/simple_client.py
# Eventually we should move all client wrappers for the message broker into a shared library that both the ingest
# service and the client can use.

import socket
import json
import time
import logging
from typing import Optional, Tuple, Union

from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema
from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase

logger = logging.getLogger(__name__)


class SimpleClient(MessageBrokerClientBase):
    """
    A client for interfacing with SimpleMessageBroker, creating a new socket connection per request
    to ensure thread safety and robustness. Respects timeouts for all operations.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        max_retries: int = 3,
        max_backoff: int = 32,
        connection_timeout: int = 300,
        max_pool_size: int = 128,
        use_ssl: bool = False,
        api_version: str = "v1",
    ):
        """
        Initialize the SimpleClient with configuration parameters.

        Parameters
        ----------
        host : str
            The hostname or IP address of the broker.
        port : int
            The port number of the broker.
        db : int, optional
            The database index (default: 0).
        max_retries : int, optional
            Maximum number of retries for operations (default: 3).
        max_backoff : int, optional
            Maximum backoff time in seconds for retries (default: 32).
        connection_timeout : int, optional
            Timeout in seconds for establishing a connection (default: 300).
        max_pool_size : int, optional
            Maximum pool size for connections (default: 128).
        use_ssl : bool, optional
            Whether to use SSL for connections (default: False).
        """

        self._host = host
        self._port = port
        self._db = db
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._max_pool_size = max_pool_size
        self._connection_timeout = connection_timeout
        self._use_ssl = use_ssl

    def get_client(self):
        """
        Retrieve the current client instance.

        Returns
        -------
        SimpleClient
            The current client instance.
        """
        return self

    def submit_message(
        self,
        queue_name: str,
        message: str,
        timeout: Optional[Tuple[int, Union[float]]] = (100, None),
        for_nv_ingest: bool = False,
    ) -> ResponseSchema:
        """
        Submit a message to the specified queue.

        Parameters
        ----------
        queue_name : str
            The name of the queue.
        message : str
            The message to be submitted.
        timeout : float, optional
            Timeout in seconds for the operation.
        for_nv_ingest : bool, optional
            Indicates whether the message is for NV ingest operations.

        Returns
        -------
        ResponseSchema
            The response from the broker.
        """
        return self._handle_push(queue_name, message, timeout, for_nv_ingest)

    def fetch_message(
        self, queue_name: str, timeout: Optional[Tuple[int, Union[float, None]]] = (1200, None)
    ) -> ResponseSchema:
        """
        Fetch a message from a specified queue.

        Parameters
        ----------
        queue_name : str
            The name of the queue.
        timeout : tuple, optional
            A tuple containing the timeout value and an unused second element.

        Returns
        -------
        ResponseSchema
            The response from the broker.
        """
        return self._handle_pop(queue_name, timeout)

    def ping(self) -> ResponseSchema:
        """
        Ping the broker to check connectivity.

        Returns
        -------
        ResponseSchema
            The response indicating the success of the ping operation.
        """
        command = {"command": "PING"}
        return self._execute_simple_command(command)

    def size(self, queue_name: str) -> ResponseSchema:
        """
        Fetch the size of the specified queue.

        Parameters
        ----------
        queue_name : str
            The name of the queue.

        Returns
        -------
        ResponseSchema
            The response containing the queue size.
        """
        command = {"command": "SIZE", "queue_name": queue_name}
        return self._execute_simple_command(command)

    def _handle_push(
        self, queue_name: str, message: str, timeout: Optional[Tuple[int, Union[float, None]]], for_nv_ingest: bool
    ) -> ResponseSchema:
        """
        Push a message to the queue with optional timeout.

        Parameters
        ----------
        queue_name : str
            The name of the queue.
        message : str
            The message to push.
        timeout : float, optional
            Timeout in seconds for the operation.
        for_nv_ingest : bool
            Indicates whether the message is for NV ingest operations.

        Returns
        -------
        ResponseSchema
            The response from the broker.
        """

        if not queue_name or not isinstance(queue_name, str):
            return ResponseSchema(response_code=1, response_reason="Invalid queue name.")
        if not message or not isinstance(message, str):
            return ResponseSchema(response_code=1, response_reason="Invalid message.")

        if for_nv_ingest:
            command = {"command": "PUSH_FOR_NV_INGEST", "queue_name": queue_name, "message": message}
        else:
            command = {"command": "PUSH", "queue_name": queue_name, "message": message}

        timeout = int(timeout[0])
        if timeout is not None:
            command["timeout"] = timeout

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            remaining_timeout = (timeout - elapsed) if (timeout is not None) else None
            if (remaining_timeout is not None) and (remaining_timeout <= 0):
                return ResponseSchema(response_code=1, response_reason="PUSH operation timed out.")

            try:
                with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                    sock.settimeout(self._connection_timeout)
                    self._send(sock, json.dumps(command).encode("utf-8"))
                    # Receive initial response with transaction ID
                    response_data = self._recv(sock)
                    response = json.loads(response_data)

                    if response.get("response_code") != 0:
                        if (
                            response.get("response_reason") == "Queue is full"
                            or response.get("response_reason") == "Queue is not available"
                        ):
                            time.sleep(0.5)
                            continue
                        else:
                            return ResponseSchema(**response)

                    if "transaction_id" not in response:
                        error_msg = "No transaction_id in response."
                        logger.error(error_msg)

                        return ResponseSchema(response_code=1, response_reason=error_msg)

                    transaction_id = response["transaction_id"]

                    # Send ACK
                    ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode("utf-8")
                    self._send(sock, ack_data)

                    # Receive final response
                    final_response_data = self._recv(sock)
                    final_response = json.loads(final_response_data)

                    return ResponseSchema(**final_response)

            except (ConnectionError, socket.error, BrokenPipeError, socket.timeout) as e:
                logger.debug(f"Connection error during PUSH: {e}")
                pass  # Will be retried
            except json.JSONDecodeError:
                return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
            except Exception as e:
                return ResponseSchema(response_code=1, response_reason=str(e))

            time.sleep(0.5)  # Backoff delay before retry

    def _handle_pop(self, queue_name: str, timeout: Optional[Tuple[int, Union[float, None]]]) -> ResponseSchema:
        """
        Pop a message from the queue with optional timeout.

        Parameters
        ----------
        queue_name : str
            The name of the queue.
        timeout : float, optional
            Timeout in seconds for the operation.

        Returns
        -------
        ResponseSchema
            The response containing the popped message.
        """

        if not queue_name or not isinstance(queue_name, str):
            return ResponseSchema(response_code=1, response_reason="Invalid queue name.")

        command = {"command": "POP", "queue_name": queue_name}

        timeout_val = timeout[0] if isinstance(timeout, tuple) else timeout

        if timeout_val is not None:
            command["timeout"] = timeout_val

        start_time = time.time()
        backoff_delay = 1  # Start with a 1-second backoff

        while True:
            elapsed = time.time() - start_time
            if timeout_val is not None and elapsed >= timeout_val:
                return ResponseSchema(response_code=2, response_reason="Job not ready.")

            try:
                with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                    sock.settimeout(self._connection_timeout)
                    self._send(sock, json.dumps(command).encode("utf-8"))
                    # Receive initial response with transaction ID and message
                    response_data = self._recv(sock)
                    response = json.loads(response_data)

                    # The broker now returns a response_code of 2 for a timeout, which the high-level
                    # client should handle as a retryable event.
                    if response.get("response_code") == 2:
                        # Queue is empty or job not ready, continue to backoff and retry
                        pass
                    elif response.get("response_code") != 0:
                        return ResponseSchema(**response)
                    else:
                        # Success case: we received a message.
                        if "transaction_id" not in response:
                            return ResponseSchema(response_code=1, response_reason="No transaction_id in response.")

                        transaction_id = response["transaction_id"]
                        message = response.get("response")

                        # Send ACK
                        ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode("utf-8")
                        self._send(sock, ack_data)

                        # Receive final response
                        final_response_data = self._recv(sock)
                        final_response = json.loads(final_response_data)

                        if final_response.get("response_code") == 0:
                            return ResponseSchema(response_code=0, response=message, transaction_id=transaction_id)
                        else:
                            return ResponseSchema(**final_response)

            except (ConnectionError, socket.error, BrokenPipeError, socket.timeout) as e:
                # Let the high-level client handle connection errors as retryable.
                logger.debug(f"Connection error during POP: {e}, will retry after backoff.")
                pass  # Fall through to backoff and retry
            except json.JSONDecodeError:
                return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
            except Exception as e:
                return ResponseSchema(response_code=1, response_reason=str(e))

            # Exponential backoff
            time.sleep(backoff_delay)
            backoff_delay = min(backoff_delay * 2, self._max_backoff)

    def _execute_simple_command(self, command: dict) -> ResponseSchema:
        """
        Execute a simple command on the broker and process the response.

        Parameters
        ----------
        command : dict
            The command to execute.

        Returns
        -------
        ResponseSchema
            The response from the broker.
        """

        if isinstance(command, dict):
            data = json.dumps(command).encode("utf-8")
        elif isinstance(command, str):
            data = command.encode("utf-8")

        try:
            with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                sock.settimeout(self._connection_timeout)
                self._send(sock, data)
                response_data = self._recv(sock)
                response = json.loads(response_data)
                return ResponseSchema(**response)
        except (ConnectionError, socket.error, BrokenPipeError, socket.timeout) as e:
            return ResponseSchema(response_code=2, response_reason=f"Connection error: {e}")
        except json.JSONDecodeError:
            return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
        except Exception as e:
            return ResponseSchema(response_code=1, response_reason=str(e))

    def _send(self, sock: socket.socket, data: bytes) -> None:
        """
        Send data over a socket connection with a length header.

        Parameters
        ----------
        sock : socket.socket
            The socket connection.
        data : bytes
            The data to send.

        Raises
        ------
        ConnectionError
            If sending data fails.
        """

        total_length = len(data)
        if total_length == 0:
            raise ValueError("Cannot send an empty message.")

        try:
            sock.sendall(total_length.to_bytes(8, "big"))
            sock.sendall(data)
        except (socket.error, BrokenPipeError):
            raise ConnectionError("Failed to send data.")

    def _recv(self, sock: socket.socket) -> str:
        """
        Receive data from a socket connection based on a length header.

        Parameters
        ----------
        sock : socket.socket
            The socket connection.

        Returns
        -------
        str
            The received data as a string.

        Raises
        ------
        ConnectionError
            If receiving data fails.
        """

        try:
            length_header = self._recv_exact(sock, 8)
            if not length_header:
                raise ConnectionError("Incomplete length header received.")
            total_length = int.from_bytes(length_header, "big")
            data_bytes = self._recv_exact(sock, total_length)
            if not data_bytes:
                raise ConnectionError("Incomplete message received.")
            return data_bytes.decode("utf-8")
        except (socket.error, BrokenPipeError, ConnectionError):
            raise ConnectionError("Failed to receive data.")

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Receive an exact number of bytes from a socket connection.

        Parameters
        ----------
        sock : socket.socket
            The socket connection.
        num_bytes : int
            The number of bytes to receive.

        Returns
        -------
        Optional[bytes]
            The received bytes, or None if the connection is closed.
        """

        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = sock.recv(num_bytes - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                return None
            except Exception:
                return None
        return bytes(data)
