# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import MagicMock
from nv_ingest_api.util.service_clients.client_base import MessageBrokerClientBase, FetchMode
from nv_ingest_api.internal.schemas.message_brokers.response_schema import ResponseSchema


def test_fetch_mode_enum_members():
    assert FetchMode.DESTRUCTIVE.name == "DESTRUCTIVE"
    assert FetchMode.NON_DESTRUCTIVE.name == "NON_DESTRUCTIVE"
    assert FetchMode.CACHE_BEFORE_DELETE.name == "CACHE_BEFORE_DELETE"


def test_fetch_mode_enum_values_are_unique():
    values = {mode.value for mode in FetchMode}
    assert len(values) == len(FetchMode)  # Ensure no duplicate values


def test_message_broker_client_base_is_abstract():
    with pytest.raises(TypeError):
        MessageBrokerClientBase(
            host="localhost",
            port=6379,
        )


class DummyMessageBrokerClient(MessageBrokerClientBase):
    """Minimal dummy concrete implementation for testing."""

    def __init__(self, *args, **kwargs):
        self.client = MagicMock()

    def get_client(self):
        return self.client

    def ping(self) -> bool:
        return True

    def fetch_message(self, job_index, timeout=(100, None), override_fetch_mode=None):
        return ResponseSchema(response_code=200, response={"job_index": job_index})

    def submit_message(self, channel_name, message, for_nv_ingest=False):
        return ResponseSchema(response_code=200, response={"channel_name": channel_name, "message": message})


def test_dummy_message_broker_client_ping_works():
    client = DummyMessageBrokerClient("localhost", 6379)
    assert client.ping() is True


def test_dummy_message_broker_client_fetch_message_works():
    client = DummyMessageBrokerClient("localhost", 6379)
    result = client.fetch_message("job-123", timeout=(50, None), override_fetch_mode=FetchMode.NON_DESTRUCTIVE)
    assert result.response_code == 200
    assert result.response["job_index"] == "job-123"
