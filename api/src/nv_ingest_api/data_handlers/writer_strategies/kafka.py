# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Kafka writer strategy for the NV-Ingest data writer.
"""

import json
from typing import List

# Type imports (will be resolved at runtime)
try:
    from nv_ingest_api.data_handlers.data_writer import KafkaDestinationConfig
except ImportError:
    # Handle circular import during development
    KafkaDestinationConfig = None


class KafkaWriterStrategy:
    """Strategy for writing to Kafka destinations."""

    def is_available(self) -> bool:
        """Check if kafka-python is available."""
        try:
            from kafka import KafkaProducer

            return KafkaProducer is not None
        except ImportError:
            return False

    def write(self, data_payload: List[str], config: KafkaDestinationConfig) -> None:
        """Write payloads to Kafka topic."""
        if not self.is_available():
            from nv_ingest_api.data_handlers.errors import DependencyError

            raise DependencyError(
                "kafka-python library is not available. Install kafka-python for Kafka destination support: "
                "pip install kafka-python"
            )

        # Import is safe now that we've checked availability
        from kafka import KafkaProducer

        # Configure producer
        producer_config = {
            "bootstrap_servers": config.bootstrap_servers,
            "security_protocol": config.security_protocol,
            "value_serializer": lambda v: (
                json.dumps(v).encode("utf-8") if config.value_serializer == "json" else str(v).encode("utf-8")
            ),
        }

        # Add SASL authentication if configured
        if config.sasl_mechanism and config.sasl_username and config.sasl_password:
            producer_config.update(
                {
                    "sasl_mechanism": config.sasl_mechanism,
                    "sasl_plain_username": config.sasl_username,
                    "sasl_plain_password": config.sasl_password,
                }
            )

        # Add SSL configuration if provided
        if config.ssl_cafile or config.ssl_certfile or config.ssl_keyfile:
            ssl_config = {}
            if config.ssl_cafile:
                ssl_config["ssl_cafile"] = config.ssl_cafile
            if config.ssl_certfile:
                ssl_config["ssl_certfile"] = config.ssl_certfile
            if config.ssl_keyfile:
                ssl_config["ssl_keyfile"] = config.ssl_keyfile
            producer_config.update(ssl_config)

        producer = None
        try:
            producer = KafkaProducer(**producer_config)

            # Send each payload as a separate message
            futures = []
            for payload in data_payload:
                data = json.loads(payload) if config.value_serializer == "json" else payload
                # Derive key based on configured key_serializer
                key = None
                if config.key_serializer == "string":
                    # If the payload is a dict and has an 'id', use it as the key
                    if isinstance(data, dict) and "id" in data and data["id"] is not None:
                        key = str(data["id"]).encode("utf-8")

                future = producer.send(config.topic, value=data, key=key)
                futures.append(future)

            # Wait for all messages to be sent
            for future in futures:
                future.get(timeout=30)  # Wait up to 30 seconds for each message

            producer.flush()  # Ensure all messages are delivered

        finally:
            if producer:
                producer.close()
