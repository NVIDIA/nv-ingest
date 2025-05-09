# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod
from enum import Enum, auto
from typing import Tuple

from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema


class FetchMode(Enum):
    DESTRUCTIVE = auto()  # Read and delete immediately (current behavior)
    NON_DESTRUCTIVE = auto()  # Read without deleting (requires TTL on Redis data)
    CACHE_BEFORE_DELETE = auto()  # Read, write to local cache, then delete from Redis


class MessageBrokerClientBase(ABC):
    """
    Abstract base class for a messaging client to interface with various messaging systems.

    Provides a standard interface for sending and receiving messages with connection management
    and retry logic.
    """

    @abstractmethod
    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        max_retries: int = 0,
        max_backoff: int = 32,
        connection_timeout: int = 300,
        max_pool_size: int = 128,
        use_ssl: bool = False,
    ):
        """
        Initialize the messaging client with connection parameters.
        """

    @abstractmethod
    def get_client(self):
        """
        Returns the client instance, reconnecting if necessary.

        Returns:
            The client instance.
        """

    @abstractmethod
    def ping(self) -> bool:
        """
        Checks if the server is responsive.

        Returns:
            True if the server responds to a ping, False otherwise.
        """

    @abstractmethod
    def fetch_message(
        self, job_index: str, timeout: Tuple[int, float] = (100, None), override_fetch_mode: FetchMode = None
    ) -> ResponseSchema:
        """
        Fetches a message from the specified queue with retries on failure.

        Parameters:
            job_index (str): The index of the job to fetch the message for.
            timeout (float): The timeout in seconds for blocking until a message is available.
            override_fetch_mode: Optional; overrides the default fetch mode.

        Returns:
            The fetched message, or None if no message could be fetched.
        """

    @abstractmethod
    def submit_message(self, channel_name: str, message: str, for_nv_ingest=False) -> ResponseSchema:
        """
        Submits a message to a specified queue with retries on failure.

        Parameters:
            channel_name (str): The name of the queue to submit the message to.
            message (str): The message to submit.
            for_nv_ingest (bool): Whether the message is for NV Ingest.
        """
