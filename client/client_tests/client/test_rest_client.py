# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nv_ingest_api.util.service_clients.rest.rest_client import RestClient


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
        http_allocator=mock_rest_client_allocator,
    )


# Test generate_url function
def test_generate_url(rest_client):
    assert rest_client._generate_url("localhost", 7670) == "http://localhost:7670"
    assert rest_client._generate_url("http://localhost", 7670) == "http://localhost:7670"
    assert rest_client._generate_url("https://localhost", 7670) == "https://localhost:7670"

    # A few more complicated and possible tricks
    assert rest_client._generate_url("localhost-https-else", 7670) == "http://localhost-https-else:7670"


# Test API version configuration
class TestApiVersionConfiguration:
    """Test suite for API version configuration and validation"""

    def test_default_api_version_is_v1(self):
        """Test that RestClient defaults to v1 when no api_version is provided"""
        client = RestClient(host="localhost", port=7670)

        assert client._api_version == "v1"
        assert client._submit_endpoint == "/v1/submit_job"
        assert client._fetch_endpoint == "/v1/fetch_job"

    def test_explicit_v2_configuration(self):
        """Test that RestClient accepts explicit v2 configuration"""
        client = RestClient(host="localhost", port=7670, api_version="v2")

        assert client._api_version == "v2"
        assert client._submit_endpoint == "/v2/submit_job"
        assert client._fetch_endpoint == "/v2/fetch_job"

    def test_invalid_version_falls_back_to_v1(self):
        """Test that invalid API versions fall back to v1 with warning"""
        client = RestClient(host="localhost", port=7670, api_version="v99")

        assert client._api_version == "v1"
        assert client._submit_endpoint == "/v1/submit_job"
        assert client._fetch_endpoint == "/v1/fetch_job"

    def test_case_normalization(self):
        """Test that API version is normalized to lowercase"""
        client = RestClient(host="localhost", port=7670, api_version="V2")

        assert client._api_version == "v2"
        assert client._submit_endpoint == "/v2/submit_job"

    def test_whitespace_normalization(self):
        """Test that API version strips whitespace"""
        client = RestClient(host="localhost", port=7670, api_version=" v2 ")

        assert client._api_version == "v2"
        assert client._submit_endpoint == "/v2/submit_job"

    @pytest.mark.parametrize("invalid_version", ["v3", "2", "api_v2", "version2", ""])
    def test_various_invalid_versions(self, invalid_version):
        """Test that various invalid versions all fall back to v1"""
        client = RestClient(host="localhost", port=7670, api_version=invalid_version)

        assert client._api_version == "v1"
        assert client._submit_endpoint == "/v1/submit_job"
