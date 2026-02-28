# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Redis writer strategy for the NV-Ingest data writer.
"""

from typing import List

# Type imports (will be resolved at runtime)
try:
    from nv_ingest_api.data_handlers.data_writer import RedisDestinationConfig
except ImportError:
    # Handle circular import during development
    RedisDestinationConfig = None


class RedisWriterStrategy:
    """Strategy for writing to Redis destinations."""

    def is_available(self) -> bool:
        """Check if Redis client is available."""
        try:
            from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

            return RedisClient is not None
        except ImportError:
            return False

    def write(self, data_payload: List[str], config: RedisDestinationConfig) -> None:
        """Write payloads to Redis message broker."""
        if not self.is_available():
            from nv_ingest_api.data_handlers.errors import DependencyError

            raise DependencyError(
                "Redis client library is not available. Install nv_ingest_api with Redis support "
                "or use a different destination type."
            )

        from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

        # Create a Redis client for this specific config
        client_kwargs = {"host": config.host, "port": config.port, "db": config.db, "password": config.password}
        try:
            redis_client = RedisClient(**client_kwargs)
        except TypeError:
            # Some environments may not support a password kwarg; retry without it
            client_kwargs.pop("password", None)
            redis_client = RedisClient(**client_kwargs)

        try:
            # Submit each payload to the channel
            for payload in data_payload:
                redis_client.submit_message(config.channel, payload)
        finally:
            # Clean up the client if needed
            pass
