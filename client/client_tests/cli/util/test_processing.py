# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from nv_ingest_client.cli.util.processing import get_valid_filename
from nv_ingest_client.cli.util.processing import save_response_data


@pytest.fixture
def text_metadata():
    return json.loads(
        r"""
        {
          "document_type": "text",
          "metadata": {
            "content": "Here is one line of text. Here is another line of text. Here is an image.",
            "source_metadata": {
              "source_name": "",
              "source_id": "\"./data/test.pdf\"",
              "source_location": "",
              "source_type": "PDF 1.4",
              "collection_id": "",
              "date_created": "2024-04-24T00:12:28.324829",
              "last_modified": "2024-04-24T00:12:28.324802",
              "summary": "",
              "partition_id": -1,
              "access_level": 1
            },
            "content_metadata": {
              "type": "text",
              "description": "Unstructured text from PDF document.",
              "page_number": -1,
              "hierarchy": {
                "page_count": 1,
                "page": -1,
                "block": -1,
                "line": -1,
                "span": -1
              }
            },
            "text_metadata": {
              "text_type": "document",
              "summary": "",
              "keywords": "",
              "language": "en"
            },
            "image_metadata": null,
            "error_metadata": null,
            "raise_on_failure": false
          }
        }
        """
    )


def test_save_response_data(tmp_path, text_metadata):
    response = {"data": [text_metadata]}

    save_response_data(response, str(tmp_path))

    assert (tmp_path / "text").is_dir()
    assert (tmp_path / "text" / "test.pdf.metadata.json").is_file()

    with open(str(tmp_path / "text" / "test.pdf.metadata.json")) as f:
        result = json.loads(f.read())

    assert result == [text_metadata]


def test_get_valid_filename():
    filename = "^&'@{}[],$=!-#()%+~_123.txt"
    assert get_valid_filename(filename) == "-_123.txt"

    with pytest.raises(ValueError) as excinfo:
        get_valid_filename("???")
        assert "Could not derive file name from '???'" in str(excinfo.value)

    # After sanitizing this would yield '..'.
    with pytest.raises(ValueError) as excinfo:
        get_valid_filename("$.$.$")
        assert "Could not derive file name from '$.$.$'" in str(excinfo.value)
