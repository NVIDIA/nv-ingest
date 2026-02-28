# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from unittest.mock import Mock, patch

from nv_ingest_api.data_handlers.writer_strategies.filesystem import FilesystemWriterStrategy
from nv_ingest_api.data_handlers.data_writer import FilesystemDestinationConfig


class TestFilesystemWriterStrategy:
    """Black box tests for FilesystemWriterStrategy."""

    def test_is_available_when_fsspec_available(self):
        """Test is_available returns True when fsspec can be imported."""
        with patch.dict("sys.modules", {"fsspec": Mock()}):
            strategy = FilesystemWriterStrategy()
            assert strategy.is_available() is True

    def test_is_available_when_fsspec_unavailable(self):
        """Test is_available returns False when fsspec cannot be imported."""
        with patch("builtins.__import__", side_effect=ImportError):
            strategy = FilesystemWriterStrategy()
            assert strategy.is_available() is False

    def test_write_success(self):
        """Test successful write to filesystem."""
        strategy = FilesystemWriterStrategy()
        config = FilesystemDestinationConfig(path="/tmp/test_output.json")

        data_payload = ['{"name": "Alice", "age": 30}', '{"name": "Bob", "age": 25}']

        # Create a proper context manager class
        class MockFileContext:
            def __init__(self):
                self.file = Mock()

            def __enter__(self):
                return self.file

            def __exit__(self, *args):
                pass

        with patch("fsspec.open", return_value=MockFileContext()) as mock_fsspec_open:
            # Should not raise any exceptions
            strategy.write(data_payload, config)

        # Verify fsspec.open was called with correct arguments
        mock_fsspec_open.assert_called_once_with("/tmp/test_output.json", "w")

    def test_write_empty_payload(self):
        """Test write with empty payload."""
        strategy = FilesystemWriterStrategy()
        config = FilesystemDestinationConfig(path="/tmp/empty.json")

        # Create a proper context manager class
        class MockFileContext:
            def __init__(self):
                self.file = Mock()

            def __enter__(self):
                return self.file

            def __exit__(self, *args):
                pass

        with patch("fsspec.open", return_value=MockFileContext()):
            strategy.write([], config)

    def test_write_dependency_error(self):
        """Test write raises DependencyError when fsspec unavailable."""
        strategy = FilesystemWriterStrategy()
        config = FilesystemDestinationConfig(path="/tmp/test.json")

        # Mock is_available to return False
        with patch.object(strategy, "is_available", return_value=False):
            from nv_ingest_api.data_handlers.errors import DependencyError

            with pytest.raises(DependencyError, match="fsspec library is not available"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_fsspec_error(self):
        """Test write handles fsspec errors."""
        strategy = FilesystemWriterStrategy()
        config = FilesystemDestinationConfig(path="/invalid/path/file.json")

        with patch("fsspec.open", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                strategy.write(['{"test": "data"}'], config)

    def test_write_json_parsing_error(self):
        """Test write handles invalid JSON in payload."""
        strategy = FilesystemWriterStrategy()
        config = FilesystemDestinationConfig(path="/tmp/test.json")

        # Invalid JSON in payload
        data_payload = ['{"valid": "json"}', "invalid json string"]

        # Should fail when trying to parse invalid JSON
        with pytest.raises(json.JSONDecodeError):
            strategy.write(data_payload, config)
