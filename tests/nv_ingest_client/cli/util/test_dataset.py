# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO

import pytest
from nv_ingest_client.cli.util.dataset import get_dataset_files
from nv_ingest_client.cli.util.dataset import get_dataset_statistics


@pytest.fixture
def dataset_content(tmp_path):
    """
    Creates a temporary JSON file with dataset content and returns its BytesIO representation.
    """
    dataset = {
        "sampled_files": [str(tmp_path / f"file{i}.txt") for i in range(5)],
        "metadata": {
            "total_sampled_size_bytes": 5000,
            "file_type_proportions": {"txt": {"target_proportion": 100, "achieved_size_bytes": 5000}},
            "sampling_method": "without_replacement",
        },
    }
    # Create temporary files to simulate dataset files
    for file_name in dataset["sampled_files"]:
        with open(file_name, "w") as f:
            f.write("Content")

    dataset_bytes = BytesIO(json.dumps(dataset).encode("utf-8"))
    return dataset_bytes


def test_get_dataset_statistics_with_valid_dataset(dataset_content):
    """
    Tests that get_dataset_statistics returns expected statistics for a valid dataset.
    """
    stats = get_dataset_statistics(dataset_content)
    assert "Dataset Statistics:" in stats
    assert "'total_number_of_files': 5" in stats


def test_get_dataset_files_without_shuffle(dataset_content):
    """
    Tests get_dataset_files returns the correct files without shuffling.
    """
    files = get_dataset_files(dataset_content, shuffle=False)
    assert len(files) == 5
    assert files[0].endswith("file0.txt")


def test_get_dataset_files_with_shuffle(dataset_content):
    """
    Tests get_dataset_files returns the correct files and shuffles them.
    """
    original_files = get_dataset_files(dataset_content, shuffle=False)
    shuffled_files = get_dataset_files(BytesIO(dataset_content.getvalue()), shuffle=True)
    assert len(shuffled_files) == 5
    assert shuffled_files != original_files, "Files should be shuffled and not match original order"


def test_get_dataset_statistics_invalid_json():
    """
    Tests get_dataset_statistics raises an error with invalid JSON.
    """
    dataset_bytes = BytesIO(b"{invalid_json}")
    with pytest.raises(json.JSONDecodeError):
        get_dataset_statistics(dataset_bytes)


def test_get_dataset_files_invalid_json():
    """
    Tests get_dataset_files raises an error with invalid JSON.
    """
    dataset_bytes = BytesIO(b"{invalid_json}")
    with pytest.raises(ValueError):
        get_dataset_files(dataset_bytes)
