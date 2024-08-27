# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock
from unittest.mock import patch

import click
import pytest
from nv_ingest_client.cli.util.click import _generate_matching_files
from nv_ingest_client.cli.util.click import click_match_and_validate_files
from nv_ingest_client.cli.util.click import click_validate_batch_size
from nv_ingest_client.cli.util.click import click_validate_file_exists
from nv_ingest_client.cli.util.click import click_validate_task
from nv_ingest_client.cli.util.click import debug_print_click_options
from nv_ingest_client.cli.util.click import pre_process_dataset
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask

_MODULE_UNDER_TEST = "nv_ingest_client.cli.util.click"


def test_click_validate_file_exists_with_existing_file(tmp_path):
    # Create a temporary file
    temp_file = tmp_path / "testfile.txt"
    temp_file.write_text("Some content")

    # Simulate Click's context and parameter.
    # (they can be None because the function doesn't use them directly in your case)
    ctx = None
    param = None
    value = str(temp_file)

    # Validate the existence of the temporary file
    assert click_validate_file_exists(ctx, param, value) == [
        value
    ], "The function should return the file path for an existing file."


def test_click_validate_file_exists_with_non_existing_file(tmp_path):
    # Define a non-existing file path
    non_existing_file = tmp_path / "non_existing_file.txt"

    # Simulate Click's context and parameter
    ctx = None
    param = None
    value = str(non_existing_file)

    # Test should raise a Click exception due to the non-existence of the file
    with pytest.raises(click.BadParameter):
        click_validate_file_exists(ctx, param, value)


def test_click_validate_file_exists_with_multiple_files(tmp_path):
    # Create multiple temporary files
    temp_file1 = tmp_path / "testfile1.txt"
    temp_file1.write_text("Some content")
    temp_file2 = tmp_path / "testfile2.txt"
    temp_file2.write_text("Some more content")

    # Simulate Click's context and parameter
    ctx = None
    param = None
    value = [str(temp_file1), str(temp_file2)]

    # Validate the existence of the temporary files
    assert (
        click_validate_file_exists(ctx, param, value) == value
    ), "The function should return the list of file paths for existing files."


# Testing validate_batch_size
def test_validate_batch_size_valid():
    assert click_validate_batch_size(None, None, 10) == 10, "Valid batch size should be returned as is"


def test_validate_batch_size_invalid():
    with pytest.raises(click.BadParameter):
        click_validate_batch_size(None, None, 0)


# Testing debug_print_click_options
@patch(f"{_MODULE_UNDER_TEST}.pprint")
def test_debug_print_click_options(mock_pprint):
    mock_ctx = Mock()
    mock_ctx.command.params = [click.Option(["--test"], is_flag=False)]
    mock_ctx.params = {"test": "value"}
    debug_print_click_options(mock_ctx)
    mock_pprint.assert_called_once_with({"test": "value"})


def test_validate_task_with_valid_split():
    """Test with valid split task options."""
    value = ['split:{"split_by": "page", "split_length": 10}']
    result = click_validate_task(None, None, value)

    assert "split" in result
    assert isinstance(result["split"], SplitTask)


def test_validate_task_with_valid_extract():
    """Test with valid extract task options."""
    value = ['extract:{"document_type": "pdf", "extract_method": "pdfium"}']
    result = click_validate_task(None, None, value)

    assert "extract_pdf" in result
    assert isinstance(result["extract_pdf"], ExtractTask)


def test_validate_task_with_valid_store_task():
    """Test with valid stor task options."""
    value = ['store:{"content_type": "image", "store_method": "minio", "endpoint": "localhost:9000"}']
    result = click_validate_task(None, None, value)

    assert "store" in result
    assert isinstance(result["store"], StoreTask)


def test_validate_task_with_invalid_task_type():
    """Test with unsupported task type."""
    value = ['unsupported:{"some_option": "value"}']
    with pytest.raises(click.BadParameter) as exc_info:
        click_validate_task(None, None, value)
    assert "Unsupported task type" in str(exc_info.value)


def test_validate_task_with_invalid_options():
    """Test validation failures due to incorrect options."""
    value = ['split:{"split_by": "unknown_method"}']
    with pytest.raises(click.BadParameter) as exc_info:
        click_validate_task(None, None, value)

    assert len(exc_info.value.args) == 1


def test_validate_task_with_incomplete_options():
    """Test validation failures due to missing required fields."""
    value = ["extract:{}"]  # Missing required 'document_type'
    with pytest.raises(click.BadParameter) as exc_info:
        click_validate_task(None, None, value)
    assert len(exc_info.value.args) == 1


@patch(f"{_MODULE_UNDER_TEST}.check_schema", side_effect=ValueError("Unsupported task type"))
def test_validate_task_with_invalid_task(mock_check_schema):
    """Test with unsupported task type."""
    value = ['unsupported:{"some_option": "value"}']
    with pytest.raises(click.BadParameter):
        click_validate_task(None, None, value)


@patch(f"{_MODULE_UNDER_TEST}.check_schema")
def test_validate_task_with_malformed_string(mock_check_schema):
    """Test with malformed task string."""
    mock_check_schema.side_effect = json.JSONDecodeError("Expecting value", "malformed_json", 0)
    value = ["split{malformed_json}"]
    with pytest.raises(click.BadParameter) as exc_info:
        click_validate_task(None, None, value)
    assert "Unsupported task type" in str(exc_info.value)


@patch(f"{_MODULE_UNDER_TEST}.check_schema")
def test_validate_task_with_json_error(mock_check_schema):
    """Test handling of JSON decode error."""
    mock_check_schema.side_effect = json.JSONDecodeError("Expecting value", "{malformed_json", 1)
    value = ['split:{"split_by": "page"']
    with pytest.raises(click.BadParameter) as exc_info:
        click_validate_task(None, None, value)

    assert len(exc_info.value.args) == 1


def create_json_file(tmp_path, content):
    """Helper function to create a temp JSON file."""
    file_path = tmp_path / "dataset.json"
    with open(file_path, "w") as f:
        json.dump(content, f)
    return str(file_path)


@pytest.fixture
def dataset(tmp_path):
    """Fixture to create a default dataset file."""
    content = {"sampled_files": ["file1.txt", "file2.txt", "file3.txt"]}
    return create_json_file(tmp_path, content)


def test_load_valid_dataset(dataset):
    """Test loading a valid dataset without shuffling."""
    files = pre_process_dataset(dataset, shuffle_dataset=False)
    assert files == ["file1.txt", "file2.txt", "file3.txt"]


def test_load_and_shuffle_dataset(tmp_path):
    """Test loading and shuffling a larger dataset."""
    # Create a larger dataset
    content = {"sampled_files": [f"file{i}.txt" for i in range(100)]}  # Creating 100 unique files
    dataset_path = create_json_file(tmp_path, content)  # Use the helper function to create the dataset file

    original_files = content["sampled_files"]
    files = pre_process_dataset(dataset_path, shuffle_dataset=True)

    assert set(files) == set(original_files), "All files should be present even if shuffled"
    assert files != original_files, "With high probability, order should be shuffled for large datasets"


def test_missing_dataset_file(tmp_path):
    """Test behavior with a non-existent dataset file."""
    non_existent_file = tmp_path / "missing.json"
    with pytest.raises(click.BadParameter):
        pre_process_dataset(str(non_existent_file), shuffle_dataset=False)


def test_malformed_json_dataset(tmp_path):
    """Test behavior with a malformed JSON file."""
    bad_json_path = tmp_path / "malformed.json"
    with open(bad_json_path, "w") as f:
        f.write("{badly: 'formatted', json: true")
    with pytest.raises(click.BadParameter):
        pre_process_dataset(str(bad_json_path), shuffle_dataset=False)


def test_empty_file_list(tmp_path):
    """Test behavior with a JSON file that contains an empty 'sampled_files' list."""
    content = {"sampled_files": []}
    file_path = create_json_file(tmp_path, content)
    files = pre_process_dataset(file_path, shuffle_dataset=True)
    assert files == [], "Expected an empty list of files"


@pytest.mark.parametrize(
    "patterns, mock_files, expected",
    [
        (["*.txt"], ["test1.txt", "test2.txt"], ["test1.txt", "test2.txt"]),
        (["*.txt"], [], []),
        (["*.md"], ["README.md"], ["README.md"]),
        (["docs/*.md"], ["docs/README.md", "docs/CHANGES.md"], ["docs/README.md", "docs/CHANGES.md"]),
    ],
)
def test_generate_matching_files(patterns, mock_files, expected):
    with patch(
        "glob.glob", side_effect=lambda pattern, recursive: [f for f in mock_files if f.startswith(pattern[:-5])]
    ), patch("os.path.isfile", return_value=True):
        assert list(_generate_matching_files(patterns)) == expected


def test_click_match_and_validate_files_found():
    with patch(f"{_MODULE_UNDER_TEST}._generate_matching_files", return_value=iter(["file1.txt", "file2.txt"])):
        result = click_match_and_validate_files(None, None, ["*.txt"])
        assert result == ["file1.txt", "file2.txt"]


def test_click_match_and_validate_files_not_found():
    with patch(f"{_MODULE_UNDER_TEST}._generate_matching_files", return_value=iter([])):
        result = click_match_and_validate_files(None, None, ["*.txt"])
        assert result == []
