# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pytest

from nv_ingest.util.tracing.tagging import traceable


class MockControlMessage:
    def __init__(self):
        self.metadata = {}
        self.timestamp = {}

    def has_metadata(self, key):
        return key in self.metadata

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def get_timestamp(self, key, default=None):
        return self.timestamp.get(key, default)

    def set_timestamp(self, key, value):
        self.timestamp[key] = value

    def filter_timestamp(self, pattern):
        return {k: v for k, v in self.timestamp.items() if pattern in k}


@pytest.fixture
def mock_control_message():
    return MockControlMessage()


# Test with trace tagging enabled and custom trace name
def test_traceable_with_trace_tagging_enabled_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable(trace_name="CustomTrace")
    def sample_function(message):
        pass  # Function body is not relevant for the test

    sample_function(mock_control_message)

    assert mock_control_message.filter_timestamp("trace::entry::CustomTrace")
    assert mock_control_message.filter_timestamp("trace::exit::CustomTrace")
    assert isinstance(mock_control_message.timestamp["trace::entry::CustomTrace"], datetime)
    assert isinstance(mock_control_message.timestamp["trace::exit::CustomTrace"], datetime)


# Test with trace tagging enabled and no custom trace name
def test_traceable_with_trace_tagging_enabled_no_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable()
    def another_function(message):
        pass  # Function body is not relevant for the test

    another_function(mock_control_message)

    assert mock_control_message.filter_timestamp("trace::entry::another_function")
    assert mock_control_message.filter_timestamp("trace::exit::another_function")
    assert isinstance(mock_control_message.timestamp["trace::entry::another_function"], datetime)
    assert isinstance(mock_control_message.timestamp["trace::exit::another_function"], datetime)


# Test with trace tagging disabled
def test_traceable_with_trace_tagging_disabled(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", False)

    @traceable()
    def disabled_function(message):
        pass  # Function body is not relevant for the test

    disabled_function(mock_control_message)

    # Ensure no trace metadata was added since trace tagging was disabled
    assert not mock_control_message.filter_timestamp("trace::")
