import select
import socket
import json
import threading
import time
import logging
from typing import Optional
from nv_ingest.util.message_brokers.simple_message_broker import ResponseSchema

logger = logging.getLogger(__name__)

import socket
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleClient:
    """
    A client for interfacing with SimpleMessageBroker, creating a new socket connection per request
    to ensure thread safety and robustness.
    """

    def __init__(self, host: str, port: int, max_retries: int = 3, max_backoff: int = 32,
                 connection_timeout: int = 300):
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout

    def submit_message(self, queue_name: str, message: str, timeout: Optional[float] = None) -> ResponseSchema:
        return self._handle_push(queue_name, message, timeout)

    def fetch_message(self, queue_name: str, timeout: Optional[float] = None) -> ResponseSchema:
        return self._handle_pop(queue_name, timeout)

    def size(self, queue_name: str) -> ResponseSchema:
        """Fetches the current number of items in the specified queue."""
        command = {"command": "SIZE", "queue_name": queue_name}
        return self._execute_command(command)

    def _execute_command(self, command: dict) -> ResponseSchema:
        if (isinstance(command, dict)):
            data = json.dumps(command).encode('utf-8')
        elif (isinstance(command, str)):
            data = command.encode('utf-8')

        retries = 0
        while retries <= self._max_retries:
            try:
                with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                    self._send(sock, data)
                    response_data = self._recv(sock)
                    logger.debug(f"RESPONSE_DATA: {response_data}")
                    response = json.loads(response_data)

                    return ResponseSchema(**response)

            except (ConnectionError, socket.error, BrokenPipeError) as e:
                logger.warning(f"Connection error: {e}, retrying.")
            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON response: {e}")
                return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
            except Exception as e:
                logger.error(f"Unexpected error during command execution: {e}", exc_info=True)
                return ResponseSchema(response_code=1, response_reason=str(e))

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying command after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="Operation failed after retries")

    def _handle_push(self, queue_name: str, message: str, timeout: Optional[float]) -> ResponseSchema:
        if not queue_name or not isinstance(queue_name, str):
            return ResponseSchema(response_code=1, response_reason="Invalid queue name.")
        if not message or not isinstance(message, str):
            return ResponseSchema(response_code=1, response_reason="Invalid message.")

        command = {"command": "PUSH", "queue_name": queue_name, "message": message}
        if timeout is not None:
            command["timeout"] = timeout

        retries = 0
        while retries <= self._max_retries:
            try:
                with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                    self._send(sock, json.dumps(command).encode('utf-8'))
                    # Receive initial response with transaction ID
                    response_data = self._recv(sock)
                    response = json.loads(response_data)
                    logger.debug(f"Initial response: {response}")

                    if 'transaction_id' not in response:
                        error_msg = "No transaction_id in response."
                        logger.error(error_msg)
                        return ResponseSchema(response_code=1, response_reason=error_msg)

                    transaction_id = response['transaction_id']

                    # Send ACK
                    ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode('utf-8')
                    self._send(sock, ack_data)
                    logger.debug(f"Sent ACK for transaction_id: {transaction_id}")

                    # Receive final response
                    final_response_data = self._recv(sock)
                    final_response = json.loads(final_response_data)
                    logger.debug(f"Final response: {final_response}")

                    return ResponseSchema(**final_response)

            except (ConnectionError, socket.error, BrokenPipeError) as e:
                logger.warning(f"Connection error during PUSH operation: {e}, retrying.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response during PUSH operation: {e}")
                return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
            except Exception as e:
                logger.error(f"Unexpected error during PUSH operation: {e}", exc_info=True)
                return ResponseSchema(response_code=1, response_reason=str(e))

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying PUSH after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="PUSH operation failed after retries")

    def _handle_pop(self, queue_name: str, timeout: Optional[float]) -> ResponseSchema:
        if not queue_name or not isinstance(queue_name, str):
            return ResponseSchema(response_code=1, response_reason="Invalid queue name.")

        command = {"command": "POP", "queue_name": queue_name}
        if timeout is not None:
            command["timeout"] = timeout

        retries = 0
        while retries <= self._max_retries:
            try:
                with socket.create_connection((self._host, self._port), timeout=self._connection_timeout) as sock:
                    self._send(sock, json.dumps(command).encode('utf-8'))
                    # Receive initial response with transaction ID and message
                    response_data = self._recv(sock)
                    response = json.loads(response_data)
                    logger.debug(f"Initial response: {response}")

                    if 'transaction_id' not in response:
                        error_msg = "No transaction_id in response."
                        logger.error(error_msg)
                        return ResponseSchema(response_code=1, response_reason=error_msg)

                    transaction_id = response['transaction_id']
                    message = response.get('response')

                    # Send ACK
                    ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode('utf-8')
                    self._send(sock, ack_data)
                    logger.debug(f"Sent ACK for transaction_id: {transaction_id}")

                    # Receive final response
                    final_response_data = self._recv(sock)
                    final_response = json.loads(final_response_data)
                    logger.debug(f"Final response: {final_response}")

                    # If the final response is successful, return the message
                    if final_response.get('response_code') == 0:
                        return ResponseSchema(response_code=0, response=message, transaction_id=transaction_id)
                    else:
                        return ResponseSchema(**final_response)

            except (ConnectionError, socket.error, BrokenPipeError) as e:
                logger.warning(f"Connection error during POP operation: {e}, retrying.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response during POP operation: {e}")
                return ResponseSchema(response_code=1, response_reason="Invalid JSON response from server.")
            except Exception as e:
                logger.error(f"Unexpected error during POP operation: {e}", exc_info=True)
                return ResponseSchema(response_code=1, response_reason=str(e))

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying POP after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="POP operation failed after retries")

    def _send(self, sock: socket.socket, data: bytes) -> None:
        """Send data with length header over the given socket."""
        total_length = len(data)

        # Ensure total length is reasonable
        if total_length == 0:
            raise ValueError("Cannot send an empty message.")

        try:
            # Send the total length of the message
            sock.sendall(total_length.to_bytes(8, 'big'))

            # Send the data
            sock.sendall(data)

            logger.debug(f"Successfully sent {total_length} bytes.")
        except (socket.error, BrokenPipeError) as e:
            logger.error(f"Socket error during send: {e}")
            raise ConnectionError("Failed to send data due to socket error.") from e

    def _recv(self, sock: socket.socket) -> str:
        """Receive data based on initial length header from the given socket."""
        try:
            # Receive the message length first
            length_header = self._recv_exact(sock, 8)
            if length_header is None:
                raise ConnectionError("Incomplete length header received.")

            total_length = int.from_bytes(length_header, 'big')

            # Validate total length
            if total_length <= 0:
                raise ValueError("Received invalid message length.")

            # Receive the message
            data_bytes = self._recv_exact(sock, total_length)
            if data_bytes is None:
                raise ConnectionError("Incomplete message received.")

            response = data_bytes.decode('utf-8', errors='replace')

            # Log the response size and preview
            logger.debug(f"Successfully received {total_length} bytes. Preview: {response[:100]}...")

            return response
        except (socket.error, BrokenPipeError, ConnectionError) as e:
            logger.error(f"Socket error during receive: {e}")
            raise ConnectionError("Failed to receive data due to socket error.") from e
        except ValueError as e:
            logger.error(f"Validation error during receive: {e}")
            raise

    def _recv_exact(self, sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Helper method to receive an exact number of bytes from the given socket."""
        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = sock.recv(num_bytes - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout as e:
                logger.error(f"Socket timeout during receive: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error during _recv_exact: {e}")
                return None
        return bytes(data)
