# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import pandas as pd
import pytest

from nv_ingest.extraction_workflows.pdf.unstructured_io_helper import unstructured_io
from nv_ingest.schemas.metadata_schema import TextTypeEnum


def requires_key():
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    return pytest.mark.skipif(api_key is None, reason="requires Unstructured.io api key")


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
            "id": ["test.pdf"],
        }
    )


@pytest.fixture
def pdf_stream():
    with open("data/test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@pytest.fixture
def table_pdf_stream():
    with open("data/table_test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@pytest.fixture
def api_key():
    return os.getenv("UNSTRUCTURED_API_KEY")


@requires_key()
@pytest.mark.parametrize(
    "text_depth",
    ["page", TextTypeEnum.PAGE],
)
def test_unstructured_io_text_depth_page(pdf_stream, document_df, text_depth, api_key):
    extracted_data = unstructured_io(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        unstructured_api_key=api_key,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    # assert extracted_data[0][0].value == "text"
    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    assert extracted_data[0][0] == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert extracted_data[0][1]["source_metadata"]["source_name"] == "test.pdf"
    assert extracted_data[0][1]["content_metadata"]["page_number"] == 0


@requires_key()
@pytest.mark.parametrize(
    "text_depth",
    ["document", TextTypeEnum.DOCUMENT],
)
def test_unstructured_io_text_depth_doc(pdf_stream, document_df, text_depth, api_key):
    extracted_data = unstructured_io(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        unstructured_api_key=api_key,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    # assert extracted_data[0][0].value == "text"
    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    assert extracted_data[0][0] == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert extracted_data[0][1]["source_metadata"]["source_name"] == "test.pdf"
    assert extracted_data[0][1]["content_metadata"]["page_number"] == -1
    assert extracted_data[0][1]["content_metadata"]["hierarchy"]["page_count"] == 1


@requires_key()
@pytest.mark.parametrize(
    "text_depth",
    ["block", TextTypeEnum.BLOCK],
)
def test_unstructured_io_text_depth_block(pdf_stream, document_df, text_depth, api_key):
    extracted_data = unstructured_io(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        unstructured_api_key=api_key,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 3
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][1]["content"] == "Here is one line of text."
    assert extracted_data[1][1]["content"] == "Here is another line of text."
    assert extracted_data[2][1]["content"] == "Here is an image."

    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert extracted_data[1][1]["source_metadata"]["source_name"] == "test.pdf"
    assert extracted_data[2][1]["content_metadata"]["page_number"] == 0


@requires_key()
def test_unstructured_io_image(pdf_stream, document_df, api_key):
    extracted_data = unstructured_io(
        pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
        unstructured_api_key=api_key,
        unstructured_strategy="hi_res",
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][0] == "image"
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"][:10] == "/9j/4AAQSk"  # JPEG format header

    assert extracted_data[1][0] == "text"
    assert (
        extracted_data[1][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[1][1]["content_metadata"]["page_number"] == 0


@requires_key()
def test_unstructured_io_table(table_pdf_stream, document_df, api_key):
    extracted_data = unstructured_io(
        table_pdf_stream,
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        row_data=document_df.iloc[0],
        unstructured_api_key=api_key,
        unstructured_strategy="hi_res",
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][0] == "structured"
    assert "<table><thead><th>Year</th><th>Bill</th><th>Amy</th><th>James</th><th>" in extracted_data[0][1]["content"]
    assert (
        "</tr><tr><td>2005</td><td></td><td>N/A</td><td>N/A</td><td>631</td><td></td></tr><tr><td>"
        in extracted_data[0][1]["content"]
    )

    assert extracted_data[0][1]["content_metadata"]["page_number"] == 0
