# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test configuration and fixtures for writer strategies tests.
"""

import pytest


@pytest.fixture
def redis_config():
    """Fixture providing a basic Redis destination configuration."""
    from nv_ingest_api.data_handlers.data_writer import RedisDestinationConfig

    return RedisDestinationConfig(host="localhost", port=6379, db=0, channel="test_channel")


@pytest.fixture
def filesystem_config(tmp_path):
    """Fixture providing a basic filesystem destination configuration."""
    from nv_ingest_api.data_handlers.data_writer import FilesystemDestinationConfig

    return FilesystemDestinationConfig(path=str(tmp_path / "test_output.json"))


@pytest.fixture
def http_config():
    """Fixture providing a basic HTTP destination configuration."""
    from nv_ingest_api.data_handlers.data_writer import HttpDestinationConfig

    return HttpDestinationConfig(url="https://api.example.com/data", method="POST")


@pytest.fixture
def kafka_config():
    """Fixture providing a basic Kafka destination configuration."""
    from nv_ingest_api.data_handlers.data_writer import KafkaDestinationConfig

    return KafkaDestinationConfig(bootstrap_servers="localhost:9092", topic="test-topic")
