import select
import socket
import json
import time
import logging
from typing import Optional
from nv_ingest.util.message_brokers.simple_message_broker import ResponseSchema

logger = logging.getLogger(__name__)


class SimpleClient:
    """
    A client for interfacing with SimpleMessageBroker, providing mechanisms for sending and receiving messages
    with retry logic, connection management, and handling of the 3-way handshake protocol.
    """

    def __init__(self, host: str, port: int, max_retries: int = 3, max_backoff: int = 32,
                 connection_timeout: int = 300):
        self._host = host
        self._port = port
        self._max_retries = max_retries
        self._max_backoff = max_backoff
        self._connection_timeout = connection_timeout
        self._socket = None  # Initialize as None to allow lazy connection

    def _connect(self) -> None:
        """Establishes a new connection to the SimpleMessageBroker server if disconnected."""
        if self._socket:
            try:
                ready = select.select([self._socket], [], [], 0)[0]
                if ready and self._socket.recv(1, socket.MSG_PEEK) == b'':
                    raise ConnectionError("Socket appears to be disconnected.")
            except (socket.error, BrokenPipeError, ConnectionError):
                self._socket.close()
                self._socket = None

        if self._socket is None:
            retries = 0
            while retries <= self._max_retries:
                try:
                    self._socket = socket.create_connection((self._host, self._port), timeout=self._connection_timeout)
                    logger.debug("Connected to SimpleMessageBroker server.")
                    return
                except socket.error as e:
                    retries += 1
                    backoff_delay = min(2 ** retries, self._max_backoff)
                    logger.warning(f"Connection attempt {retries} failed: {e}. Retrying in {backoff_delay} seconds.")
                    time.sleep(backoff_delay)

            raise ConnectionError("Failed to connect to SimpleMessageBroker server after retries.")

    def _send(self, data: bytes) -> None:
        """Send data with length header."""
        try:
            if not self._socket:
                raise ConnectionError("Socket is not connected.")

            total_length = len(data)

            # Ensure total length is reasonable
            if total_length == 0:
                raise ValueError("Cannot send an empty message.")
            if total_length > 2 ** 63 - 1:
                raise ValueError("Message size exceeds allowable limit.")

            # Send the total length of the message
            self._socket.sendall(total_length.to_bytes(8, 'big'))

            # Send the data
            self._socket.sendall(data)

            logger.debug(f"Successfully sent {total_length} bytes.")
        except (socket.error, BrokenPipeError, ConnectionError) as e:
            logger.error(f"Socket error during send: {e}")
            if self._socket:
                self._socket.close()
                self._socket = None
            raise ConnectionError("Failed to send data due to socket error.") from e
        except ValueError as e:
            logger.error(f"Validation error during send: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during send: {e}")
            if self._socket:
                self._socket.close()
                self._socket = None
            raise

    def _recv(self) -> str:
        """Receive data based on initial length header."""
        try:
            if not self._socket:
                raise ConnectionError("Socket is not connected.")

            # Receive the message length first
            length_header = self._recv_exact(8)
            if length_header is None:
                raise ConnectionError("Incomplete length header received.")

            total_length = int.from_bytes(length_header, 'big')

            # Validate total length
            if total_length <= 0:
                raise ValueError("Received invalid message length.")
            if total_length > 2 ** 63 - 1:
                raise ValueError("Message length exceeds allowable limit.")

            # Receive the message
            data_bytes = self._recv_exact(total_length)
            if data_bytes is None:
                raise ConnectionError("Incomplete message received.")

            response = data_bytes.decode('utf-8')

            # Log the response size and preview
            logger.debug(f"Successfully received {total_length} bytes. Preview: {response[:100]}...")

            return response
        except (socket.error, BrokenPipeError, ConnectionError) as e:
            logger.error(f"Socket error during receive: {e}")
            if self._socket:
                self._socket.close()
                self._socket = None
            raise ConnectionError("Failed to receive data due to socket error.") from e
        except ValueError as e:
            logger.error(f"Validation error during receive: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during receive: {e}")
            if self._socket:
                self._socket.close()
                self._socket = None
            raise

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """Helper method to receive an exact number of bytes."""
        data = bytearray()
        while len(data) < num_bytes:
            packet = self._socket.recv(num_bytes - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def submit_message(self, queue_name: str, message: str, timeout: Optional[float] = None) -> ResponseSchema:
        return self._handle_push(queue_name, message, timeout)

    def fetch_message(self, queue_name: str, timeout: Optional[float] = None) -> ResponseSchema:
        return self._handle_pop(queue_name, timeout)

    def size(self, queue_name: str) -> ResponseSchema:
        """Fetches the current number of items in the specified queue."""
        data = json.dumps({"command": "SIZE", "queue_name": queue_name})
        return self._execute_command(data.encode('utf-8'))

    def _execute_command(self, data: bytes) -> ResponseSchema:
        retries = 0
        while retries <= self._max_retries:
            try:
                self._connect()
                self._send(data)
                response_data = self._recv()
                logger.debug(f"RESPONSE_DATA: {response_data}")
                response = json.loads(response_data)

                return ResponseSchema(**response)

            except (ConnectionError, BrokenPipeError) as e:
                logger.warning(f"Connection error: {e}, retrying connection.")
                if self._socket:
                    self._socket.close()
                    self._socket = None
            except json.JSONDecodeError as e:
                logger.error(f"Received invalid JSON response: {e}")
                # Retry on JSON decode error
            except Exception as e:
                logger.error(f"Unexpected error during command execution: {e}")
                # Retry on unexpected exceptions

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying command after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="Operation failed after retries")

    def _handle_push(self, queue_name: str, message: str, timeout: Optional[float]) -> ResponseSchema:
        command = {"command": "PUSH", "queue_name": queue_name, "message": message}
        if timeout is not None:
            command["timeout"] = timeout
        command_data = json.dumps(command).encode('utf-8')
        retries = 0
        while retries <= self._max_retries:
            try:
                self._connect()
                self._send(command_data)
                # Receive initial response with transaction ID
                response_data = self._recv()
                response = json.loads(response_data)
                logger.debug(f"Initial response: {response}")

                if 'transaction_id' not in response:
                    logger.error("No transaction_id in response.")
                    raise ValueError("No transaction_id in response")

                transaction_id = response['transaction_id']

                # Send ACK
                ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode('utf-8')
                self._send(ack_data)
                logger.debug(f"Sent ACK for transaction_id: {transaction_id}")

                # Receive final response
                final_response_data = self._recv()
                final_response = json.loads(final_response_data)
                logger.debug(f"Final response: {final_response}")

                return ResponseSchema(**final_response)

            except (ConnectionError, BrokenPipeError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error during PUSH operation: {e}, retrying.")
                if self._socket:
                    self._socket.close()
                    self._socket = None
            except Exception as e:
                logger.error(f"Unexpected error during PUSH operation: {e}")
                if self._socket:
                    self._socket.close()
                    self._socket = None
                # Decide whether to retry or not
                raise

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying PUSH after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="PUSH operation failed after retries")

    def _handle_pop(self, queue_name: str, timeout: Optional[float]) -> ResponseSchema:
        command = {"command": "POP", "queue_name": queue_name}
        if timeout is not None:
            command["timeout"] = timeout
        command_data = json.dumps(command).encode('utf-8')
        retries = 0
        while retries <= self._max_retries:
            try:
                self._connect()
                self._send(command_data)
                # Receive initial response with transaction ID and message
                response_data = self._recv()
                response = json.loads(response_data)
                logger.debug(f"Initial response: {response}")

                if 'transaction_id' not in response:
                    logger.error("No transaction_id in response.")
                    raise ValueError("No transaction_id in response")

                transaction_id = response['transaction_id']
                message = response.get('response')

                # Send ACK
                ack_data = json.dumps({"transaction_id": transaction_id, "ack": True}).encode('utf-8')
                self._send(ack_data)
                logger.debug(f"Sent ACK for transaction_id: {transaction_id}")

                # Receive final response
                final_response_data = self._recv()
                final_response = json.loads(final_response_data)
                logger.debug(f"Final response: {final_response}")

                # If the final response is successful, return the message
                if final_response.get('response_code') == 0:
                    return ResponseSchema(response_code=0, response=message, transaction_id=transaction_id)
                else:
                    return ResponseSchema(**final_response)

            except (ConnectionError, BrokenPipeError, json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error during POP operation: {e}, retrying.")
                if self._socket:
                    self._socket.close()
                    self._socket = None
            except Exception as e:
                logger.error(f"Unexpected error during POP operation: {e}")
                if self._socket:
                    self._socket.close()
                    self._socket = None
                # Decide whether to retry or not
                raise

            retries += 1
            backoff_delay = min(2 ** retries, self._max_backoff)
            logger.warning(f"Retrying POP after error, backoff delay: {backoff_delay}s")
            time.sleep(backoff_delay)

        return ResponseSchema(response_code=1, response_reason="POP operation failed after retries")