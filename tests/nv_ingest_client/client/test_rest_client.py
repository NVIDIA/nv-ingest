# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nv_ingest_client.message_clients.rest.rest_client import RestClient


class MockRestClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.counter = 0

    def get_client(self):
        return self


@pytest.fixture
def mock_rest_client_allocator():
    return MagicMock(return_value=MockRestClient("localhost", 7670))


@pytest.fixture
def rest_client(mock_rest_client_allocator):
    return RestClient(
        host="localhost",
        port=7670,
        max_retries=0,
        max_backoff=32,
        connection_timeout=300,
        http_allocator=mock_rest_client_allocator
    )


# Test generate_url function
def test_generate_url(rest_client):
    assert rest_client.generate_url("localhost", 7670) == "http://localhost:7670"
    assert rest_client.generate_url("http://localhost", 7670) == "http://localhost:7670"
    assert rest_client.generate_url("https://localhost", 7670) == "https://localhost:7670"
    
    # A few more complicated and possible tricks
    assert rest_client.generate_url("localhost-https-else", 7670) == "http://localhost-https-else:7670"
