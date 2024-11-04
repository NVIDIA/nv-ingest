# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from io import BytesIO
from io import StringIO

import pandas as pd
import pytest

from nv_ingest.extraction_workflows.pdf.pdfium_helper import pdfium_extractor
from nv_ingest.schemas.metadata_schema import TextTypeEnum


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
        }
    )


@pytest.fixture
def pdf_stream_test_pdf():
    with open("data/test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@pytest.fixture
def pdf_stream_embedded_tables_pdf():
    with open("data/embedded_table.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@pytest.mark.xfail(reason="PDFium conversion required")
def test_pdfium_extractor_basic(pdf_stream_test_pdf, document_df):
    extracted_data = pdfium_extractor(
        pdf_stream_test_pdf,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"


@pytest.mark.xfail(reason="PDFium does not support span line and block level extraction")
@pytest.mark.parametrize(
    "text_depth",
    ["span", TextTypeEnum.SPAN, "line", TextTypeEnum.LINE, "block", TextTypeEnum.BLOCK],
)
def test_pdfium_extractor_text_depth_line(pdf_stream_test_pdf, document_df, text_depth):
    extracted_data = pdfium_extractor(
        pdf_stream_test_pdf,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 3
    assert all(len(x) == 3 for x in extracted_data)
    assert all(x[0].value == "text" for x in extracted_data)
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"] == "Here is one line of text."
    assert extracted_data[1][1]["content"] == "Here is another line of text."
    assert extracted_data[2][1]["content"] == "Here is an image."
    assert all(x[1]["source_metadata"]["source_id"] == "source1" for x in extracted_data)


@pytest.mark.xfail(reason="PDFium does not support span line and block level extraction")
@pytest.mark.parametrize(
    "text_depth",
    ["page", TextTypeEnum.PAGE, "document", TextTypeEnum.DOCUMENT],
)
def test_pdfium_extractor_text_depth_page(pdf_stream_test_pdf, document_df, text_depth):
    extracted_data = pdfium_extractor(
        pdf_stream_test_pdf,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "text"
    assert isinstance(extracted_data[0][2], str)
    assert (
        extracted_data[0][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"


@pytest.mark.xfail(reason="PDFium conversion required")
def test_pdfium_extractor_extract_image(pdf_stream_test_pdf, document_df):
    extracted_data = pdfium_extractor(
        pdf_stream_test_pdf,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0].value == "image"
    assert all(isinstance(x[2], str) for x in extracted_data)
    assert extracted_data[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
    assert extracted_data[1][0].value == "text"
    assert (
        extracted_data[1][1]["content"] == "Here is one line of text. Here is another line of text. Here is an image."
    )


@pytest.mark.xfail(reason="PDFium conversion required")
def read_markdown_table(table_str: str) -> pd.DataFrame:
    """Read markdown table from string and return pandas DataFrame."""
    # Ref: https://stackoverflow.com/a/76184953/
    cleaned_table_str = re.sub(r"(?<=\|)( *[\S ]*? *)(?=\|)", lambda match: match.group(0).strip(), table_str)
    df = (
        pd.read_table(StringIO(cleaned_table_str), sep="|", header=0, skipinitialspace=True)
        .dropna(axis=1, how="all")
        .iloc[1:]
    )
    df.columns = df.columns.str.strip()
    return df


@pytest.mark.xfail(reason="PDFium conversion required")
def test_pdfium_extractor_table_extraction_on_pdf_with_no_tables(pdf_stream_test_pdf, document_df):
    extracted_data = pdfium_extractor(
        pdf_stream_test_pdf,
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        extract_tables_method="pdfium",
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 0


@pytest.mark.xfail(reason="PDFium conversion required")
def test_pdfium_extractor_table_extraction_on_pdf_with_tables(pdf_stream_embedded_tables_pdf, document_df):
    """
    Test to ensure pdfium's table extraction is able to extract easy-to-read tables from a PDF.
    """
    extracted_data = pdfium_extractor(
        pdf_stream_embedded_tables_pdf,
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        extract_tables_method="pdfium",
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 8
    assert all(len(x) == 3 for x in extracted_data)
    assert all(x[0].value == "structured" for x in extracted_data)
    assert [x[1]["content_metadata"]["page_number"] for x in extracted_data] == [0, 0, 0, 1, 1, 1, 1, 1]

    tables_markdown_format = [x[1]["content"] for x in extracted_data]
    dfs = [read_markdown_table(table_markdown_format) for table_markdown_format in tables_markdown_format]

    # Computation table
    assert list(dfs[0].columns) == ["Depen- dency", "Minimum Ver- sion", "Notes"]
    assert any(["Alternative execution engine for rolling operations" in row for row in dfs[0]["Notes"].values])

    # Excel files table
    assert list(dfs[1].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[1]["Dependency"].to_list() == ["xlrd", "xlwt", "xlsxwriter", "openpyxl", "pyxlsb"]

    # HTML table
    assert list(dfs[2].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[2]["Dependency"].to_list() == ["BeautifulSoup4", "html5lib", "lxml"]

    # XML table
    assert list(dfs[3].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[3]["Dependency"].to_list() == ["lxml"]

    # SQL databases table
    assert list(dfs[4].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[4]["Dependency"].to_list() == ["SQLAlchemy", "psycopg2", "pymysql"]

    # Other data sources table
    assert list(dfs[5].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[5]["Dependency"].to_list() == ["PyTables", "blosc", "zlib", "fastparquet", "pyarrow", "pyreadstat"]

    # Warning table
    assert list(dfs[6].columns) == ["System", "Conda", "PyPI"]
    assert dfs[6]["System"].to_list() == ["Linux", "macOS", "Windows"]

    # Access data in the cloud table
    assert list(dfs[7].columns) == ["Dependency", "Minimum Version", "Notes"]
    assert dfs[7]["Dependency"].to_list() == ["fsspec", "gcsfs", "pandas-gbq", "s3fs"]
