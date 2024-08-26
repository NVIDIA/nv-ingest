# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import stat
from unittest.mock import patch

import pytest
from nv_ingest_client.cli.util.system import configure_logging
from nv_ingest_client.cli.util.system import ensure_directory_with_permissions
from nv_ingest_client.cli.util.system import has_permissions

_MODULE_UNDER_TEST = "nv_ingest_client.cli.util.system"


@pytest.fixture
def setup_files(tmp_path):
    readable_file = tmp_path / "readable.txt"
    writable_file = tmp_path / "writable.txt"
    readable_file.write_text("readable content")
    writable_file.write_text("writable content")
    # Setting file permissions, note: os.chmod may behave differently across different OS
    readable_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    writable_file.chmod(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    return readable_file, writable_file


def test_ensure_directory_with_permissions_create(tmp_path):
    new_dir = tmp_path / "new_dir"
    ensure_directory_with_permissions(str(new_dir)), "Should create directory and return True"


def test_ensure_directory_with_permissions_existing(tmp_path):
    # Using an existing directory (tmp_path itself) to check read/write permission
    ensure_directory_with_permissions(str(tmp_path)), "Should return True for existing directory with permissions"


def test_ensure_directory_with_permissions_no_write_permission_parent(tmp_path, monkeypatch):
    # Temporarily modify has_permissions to simulate no write permission on parent directory
    def mock_has_permissions(path: str, read: bool = False, write: bool = False) -> bool:
        if write:
            return False  # Simulate no write permission
        return True  # Assume read permission is okay

    monkeypatch.setattr("nv_ingest_client.cli.util.system.has_permissions", mock_has_permissions)

    with pytest.raises(OSError) as e:
        ensure_directory_with_permissions(str(tmp_path / "no_write_permission"))
    assert "does not have write permissions" in str(
        e.value
    ), "Should raise an OSError due to no write permission on parent"


# Testing configure_logging
@patch("logging.Logger.setLevel")
@patch("logging.basicConfig")
def test_configure_logging(mock_basicConfig, mock_setLevel):
    logger = logging.getLogger(__name__)
    configure_logging(logger, "DEBUG")
    mock_basicConfig.assert_called_with(level=logging.DEBUG)
    mock_setLevel.assert_called_with(logging.DEBUG)

    with pytest.raises(ValueError):
        configure_logging(logger, "INVALID")


def test_has_permissions_nonexistent_path(tmp_path):
    nonexistent_path = tmp_path / "nonexistent"
    assert not has_permissions(str(nonexistent_path)), "Nonexistent path should return False"


def test_has_permissions_readable_file(setup_files):
    readable_file, _ = setup_files
    assert has_permissions(str(readable_file), read=True), "Should return True for readable file"


def test_has_permissions_writable_file(setup_files):
    _, writable_file = setup_files
    assert has_permissions(str(writable_file), write=True), "Should return True for writable file"
