# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from nv_ingest.util.tracing.latency import latency_logger

MODULE_UNDER_TEST = "nv_ingest.util.tracing.latency"


class MockControlMessage:
    def __init__(self):
        self.metadata = {}
        self.timestamp = {}

    def has_metadata(self, key):
        return key in self.metadata

    def get_metadata(self, key):
        return self.metadata.get(key, None)

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def get_timestamp(self, key, default=None):
        return self.timestamp.get(key, default)

    def set_timestamp(self, key, value):
        self.timestamp[key] = value

    def filter_timestamp(self, pattern):
        return {k: v for k, v in self.timestamp.items() if pattern in k}


# Mocked function to be decorated
def test_function(control_message):
    return "Test Function Executed"


# Decorating the test function
decorated_test_function = latency_logger()(test_function)

# Decorating with custom name
decorated_test_function_custom_name = latency_logger(name="CustomName")(test_function)


@pytest.fixture
def control_message():
    return MockControlMessage()


@patch(f"{MODULE_UNDER_TEST}.logging")
def test_latency_logger_without_existing_metadata(mock_logging, control_message):
    result = decorated_test_function(control_message)

    assert result == "Test Function Executed"
    assert not mock_logging.debug.called  # No existing ts_send, no log about "since ts_send"
    assert control_message.filter_timestamp("latency::test_function::elapsed_time")


def test_latency_logger_with_custom_name(control_message):
    # Custom name test without patching logging to simply test metadata setting
    decorated_test_function_custom_name(control_message)

    assert control_message.filter_timestamp("latency::CustomName::elapsed_time")


def test_latency_logger_with_invalid_argument():
    with pytest.raises(ValueError):
        # Passing an object without metadata handling capabilities
        decorated_test_function("invalid_argument")
