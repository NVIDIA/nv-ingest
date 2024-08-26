# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import pandas as pd
import pytest

from nv_ingest.schemas.metadata_schema import TextTypeEnum

from ....import_checks import ADOBE_IMPORT_OK

if ADOBE_IMPORT_OK:
    from nv_ingest.extraction_workflows.pdf.adobe_helper import adobe


def requires_sdk_client_id_and_secret():
    client_id = os.getenv("ADOBE_CLIENT_ID")
    client_secret = os.getenv("ADOBE_CLIENT_SECRET")

    reqs_ok = (client_id is not None) and (client_secret is not None) and (ADOBE_IMPORT_OK)
    print(reqs_ok)

    return pytest.mark.skipif(not reqs_ok, reason="requires Adobe client id and secret")


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
def client_id():
    return os.getenv("ADOBE_CLIENT_ID")


@pytest.fixture
def client_secret():
    return os.getenv("ADOBE_CLIENT_SECRET")


@requires_sdk_client_id_and_secret()
@pytest.mark.parametrize(
    "text_depth",
    ["page", TextTypeEnum.PAGE],
)
def test_adobe_text_depth_page(pdf_stream, document_df, text_depth, client_id, client_secret):
    extracted_data = adobe(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        adobe_client_id=client_id,
        adobe_client_secret=client_secret,
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


@requires_sdk_client_id_and_secret()
@pytest.mark.parametrize(
    "text_depth",
    ["document", TextTypeEnum.DOCUMENT],
)
def test_adobe_text_depth_doc(pdf_stream, document_df, text_depth, client_id, client_secret):
    extracted_data = adobe(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        adobe_client_id=client_id,
        adobe_client_secret=client_secret,
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


@requires_sdk_client_id_and_secret()
@pytest.mark.parametrize(
    "text_depth",
    ["block", TextTypeEnum.BLOCK],
)
def test_adobe_text_depth_block(pdf_stream, document_df, text_depth, client_id, client_secret):
    extracted_data = adobe(
        pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
        adobe_client_id=client_id,
        adobe_client_secret=client_secret,
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


@requires_sdk_client_id_and_secret()
def test_adobe_image(pdf_stream, document_df, client_id, client_secret):
    extracted_data = adobe(
        pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
        adobe_client_id=client_id,
        adobe_client_secret=client_secret,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][0] == "image"
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header

    assert extracted_data[1][0] == "text"
    assert (
        extracted_data[1][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[1][1]["content_metadata"]["page_number"] == 0


@requires_sdk_client_id_and_secret()
def test_adobe_table(table_pdf_stream, document_df, client_id, client_secret):
    extracted_data = adobe(
        table_pdf_stream,
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        row_data=document_df.iloc[0],
        adobe_client_id=client_id,
        adobe_client_secret=client_secret,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][0] == "structured"

    print(extracted_data)
    assert (
        "Bill" in extracted_data[0][1]["content"]
        and "Amy" in extracted_data[0][1]["content"]
        and "James" in extracted_data[0][1]["content"]
        and "Ted" in extracted_data[0][1]["content"]
        and "Susan" in extracted_data[0][1]["content"]
        and "N/A" in extracted_data[0][1]["content"]
        and all([str(year) in extracted_data[0][1]["content"] for year in range(2004, 2024)])
    )
    assert extracted_data[0][1]["content_metadata"]["page_number"] == 0
