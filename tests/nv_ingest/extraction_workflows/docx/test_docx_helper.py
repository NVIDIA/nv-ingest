# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from io import BytesIO

import pandas as pd
import pytest

from nv_ingest.extraction_workflows.docx.docx_helper import python_docx
from nv_ingest.schemas.metadata_schema import ImageTypeEnum


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["woods_frost"],
        }
    )


@pytest.fixture
def doc_stream():
    with open("data/woods_frost.docx", "rb") as f:
        doc_stream = BytesIO(f.read())
    return doc_stream


def test_docx_all_text(doc_stream, document_df):
    """
    Validate text extraction
    """
    extracted_data = python_docx(
        doc_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    expected_text_cnt = 1
    # data per entry
    expected_col_cnt = 3
    expected_content = "## Stopping by Woods on a Snowy Evening, By Robert Frost\n <image 1>\n *Figure 1: Snowy Woods*\n Whose woods these are I think I know. His house is in the village though; He will not see me stopping here; To watch his woods fill up with snow.\n \n My little horse must think it queer; To stop without a farmhouse near; Between the woods and frozen lake; The darkest evening of the year.\n \n He gives his harness bells a shake; To ask if there is some mistake.\u00a0The only other sound\u2019s the sweep; Of easy wind and downy flake.\n \n The woods are lovely, dark and deep, But I have promises to keep,\u00a0And miles to go before I sleep,\u00a0\u00a0And miles to go before I sleep.\n \n ## Frost\u2019s Collections\n <image 2>\n *Figure 2: Robert Frost*\n \n \n".splitlines()  # noqa: E501

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == expected_text_cnt
    assert len(extracted_data[0]) == expected_col_cnt

    # validate parsed content type
    # assert extracted_data[0][0].value == "text"
    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    assert extracted_data[0][0] == "text"
    assert isinstance(extracted_data[0][2], str)

    # validate parsed text
    extracted_content = extracted_data[0][1]["content"].splitlines()
    assert len(extracted_content) == len(expected_content)
    assert extracted_content == expected_content

    # validate document name
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "woods_frost"


@pytest.mark.xfail(reason="Table extract requires yolox, disabling for now")
def test_docx_table(doc_stream, document_df):
    """
    Validate text and table extraction. Table content is converted into markdown text.
    """
    extracted_data = python_docx(
        doc_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    expected_text_cnt = 1
    # data per entry
    expected_col_cnt = 3
    expected_content = "## Stopping by Woods on a Snowy Evening, By Robert Frost\n <image 1>\n *Figure 1: Snowy Woods*\n Whose woods these are I think I know. His house is in the village though; He will not see me stopping here; To watch his woods fill up with snow.\n \n My little horse must think it queer; To stop without a farmhouse near; Between the woods and frozen lake; The darkest evening of the year.\n \n He gives his harness bells a shake; To ask if there is some mistake.\u00a0The only other sound\u2019s the sweep; Of easy wind and downy flake.\n \n The woods are lovely, dark and deep, But I have promises to keep,\u00a0And miles to go before I sleep,\u00a0\u00a0And miles to go before I sleep.\n \n ## Frost\u2019s Collections\n <image 2>\n *Figure 2: Robert Frost*\n \n |   # | Collection         | Year    |\n|----:|:-------------------|:--------|\n|   1 | A Boy's Will       | 1913    |\n|   2 | North of Boston    | 1914    |\n|   3 | Mountain Interval  | 1916    |\n|   4 | New Hampshire      | 1923    |\n|   5 | West Running Brook | 1928    |\n|   6 | A Further Range    | 1937    |\n|   7 | A Witness Tree     | 1942    |\n|   8 | In the Clearing    | 1962    |\n|   9 | Steeple Bush       | 1947    |\n|  10 | An Afterword       | unknown |\n \n".splitlines()  # noqa: E501

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == expected_text_cnt
    assert len(extracted_data[0]) == expected_col_cnt

    # validate parsed content type
    # assert extracted_data[0][0].value == "text"
    # Work around until https://github.com/apache/arrow/pull/40412 is resolved
    assert extracted_data[0][0] == "text"
    assert isinstance(extracted_data[0][2], str)

    # validate parsed text
    extracted_content = extracted_data[0][1]["content"].splitlines()
    assert len(extracted_content) == len(expected_content)
    assert extracted_content == expected_content

    # validate document name
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "woods_frost"


def test_docx_image(doc_stream, document_df):
    """
    Validate text and table extraction. Table content is converted into markdown text.
    """
    extracted_data = python_docx(
        doc_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    expected_text_cnt = 1
    expected_image_cnt = 2
    expected_entry_cnt = expected_image_cnt + expected_text_cnt
    # data per entry
    expected_col_cnt = 3

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == expected_entry_cnt
    assert len(extracted_data[0]) == expected_col_cnt

    # two images are extracted from the test document
    image_cnt = 0
    for idx in range(expected_image_cnt):
        if extracted_data[idx][0] == "text":
            continue

        image_cnt += 1
        assert extracted_data[idx][0] == "image"

        # validate image type
        assert extracted_data[idx][1]["image_metadata"]["image_type"] == ImageTypeEnum.image_type_1
