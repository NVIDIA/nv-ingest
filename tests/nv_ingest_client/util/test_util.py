# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
