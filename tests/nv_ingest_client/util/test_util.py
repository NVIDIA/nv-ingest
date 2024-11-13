# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from nv_ingest_client.cli.util.click import generate_matching_files
from nv_ingest_client.util.util import filter_function_kwargs

_MODULE_UNDER_TEST = "nv_ingest_client.util.util"


# @pytest.mark.parametrize(
#    "file_path,page_count",
#    [
#        ("/fake/path/document.pdf", 10),  # Assume PDF has 10 pages
#        (
#            "/fake/path/document.txt",
#            3,
#        ),  # Assume text file corresponds to 3 pages (900 words)
#        (
#            "/fake/path/document.html",
#            2,
#        ),  # Assume Markdown file corresponds to 2 pages (600 words)
#        (
#            "/fake/path/document.md",
#            5,
#        ),  # Assume Markdown file corresponds to 5 pages (1500 words)
#    ],
# )
# def test_estimate_page_count_supported_types(file_path, page_count):
#    """
#    Test that estimate_page_count returns the correct page count for supported file types.
#    """
#    if file_path.endswith(".pdf"):
#        with patch(f"{_MODULE_UNDER_TEST}.fitz.open") as mock_fitz:
#            mock_doc = mock_fitz.return_value.__enter__.return_value
#            mock_doc.__len__.return_value = page_count
#            assert estimate_page_count(file_path) == page_count
#    else:
#        # Simulate binary content of a text or markdown file
#        words = "word " * 300 * page_count
#        mock_file_read_data = BytesIO(words.encode("utf-8"))
#
#        # Patch 'open' to return a mock that simulates 'file.read()' for binary content
#        with patch("builtins.open", MagicMock(return_value=mock_file_read_data)):
#            # Additionally, patch 'detect_encoding_and_read_text_file' to directly return the words
#            with patch(
#                f"{_MODULE_UNDER_TEST}.detect_encoding_and_read_text_file",
#                return_value=words,
#            ):
#                assert estimate_page_count(file_path) == page_count


# def test_estimate_page_count_file_not_found():
#     """
#     Test that estimate_page_count returns None when the file does not exist.
#     """
#     file_path = "/fake/path/nonexistent.pdf"
#     with patch(f"{_MODULE_UNDER_TEST}.os.path.splitext", return_value=("", ".pdf")):
#         with patch(f"{_MODULE_UNDER_TEST}.fitz.open", side_effect=FileNotFoundError):
#             assert estimate_page_count(file_path) == 0


# def test_estimate_page_count_general_exception():
#     """
#     Test that estimate_page_count returns None for any general exception.
#     """
#     file_path = "/fake/path/problematic.pdf"
#     with patch(f"{_MODULE_UNDER_TEST}.os.path.splitext", return_value=("", ".pdf")):
#         with patch(f"{_MODULE_UNDER_TEST}.fitz.open", side_effect=Exception("Some error")):
#             assert estimate_page_count(file_path) == 0


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
        assert list(generate_matching_files(patterns)) == expected


def test_filter_function_kwargs_with_matching_kwargs():
    def sample_func(a, b, c):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == {"a": 1, "b": 2, "c": 3}, "Should only include kwargs matching the function parameters"


def test_filter_function_kwargs_with_no_matching_kwargs():
    def sample_func(a, b, c):
        pass

    kwargs = {"x": 10, "y": 20}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == {}, "Should return an empty dictionary when there are no matching kwargs"


def test_filter_function_kwargs_with_partial_matching_kwargs():
    def sample_func(a, b):
        pass

    kwargs = {"a": 1, "x": 99, "y": 42}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == {"a": 1}, "Should include only kwargs that match the function's parameters"


def test_filter_function_kwargs_with_no_kwargs():
    def sample_func(a, b):
        pass

    kwargs = {}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == {}, "Should return an empty dictionary when no kwargs are provided"


def test_filter_function_kwargs_with_extra_kwargs_ignored():
    """Test that extra kwargs are ignored and do not cause errors."""

    def sample_func(a, b):
        pass

    kwargs = {"a": 10, "b": 20, "extra": "ignored"}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == {"a": 10, "b": 20}, "Should ignore extra kwargs not in the function parameters"


def test_filter_function_kwargs_all_args_matching():
    """Test that all kwargs are returned if they all match the function parameters."""

    def sample_func(a, b, c):
        pass

    kwargs = {"a": 5, "b": 10, "c": 15}
    result = filter_function_kwargs(sample_func, **kwargs)
    assert result == kwargs, "Should return all kwargs when they match all function parameters"
