# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from io import BytesIO
from textwrap import dedent

import numpy
import pandas as pd
import pytest
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

from nv_ingest.extraction_workflows.pptx.pptx_helper import python_pptx


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
        }
    )


@pytest.fixture
def pptx_stream_with_text():
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]

    title.text = "Hello, World!"
    subtitle.text = "This is a subtitle."

    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)

    left = top = width = height = Inches(1)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame

    tf.text = "This is text inside a textbox"

    p = tf.add_paragraph()
    p.text = "This is a second paragraph that's bold"
    p.font.bold = True

    p = tf.add_paragraph()
    p.text = "This is a third paragraph that's italic"
    p.font.italic = True
    r = p.add_run()
    r.text = ""  # testing an empty run

    p = tf.add_paragraph()
    p.text = ""  # testing an empty text

    p = tf.add_paragraph()
    p.text = "This is a fourth paragraph that's underlined"
    p.font.underline = True

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = "link to NVIDIA"
    r.hyperlink.address = "https://www.nvidia.com/en-us/"

    pptx_stream = BytesIO()
    prs.save(pptx_stream)
    return pptx_stream


@pytest.fixture
def pptx_stream_with_multiple_runs_in_title():
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]

    p = title.text_frame.paragraphs[0]
    run1 = p.add_run()
    run1.text = "Hello, "
    run2 = p.add_run()
    run2.text = "World!"

    p = subtitle.text_frame.paragraphs[0]
    run1 = p.add_run()
    run1.text = "Subtitle "
    run2 = p.add_run()
    run2.text = "here."

    pptx_stream = BytesIO()
    prs.save(pptx_stream)
    return pptx_stream


@pytest.fixture
def pptx_stream_with_group():
    prs = Presentation()

    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)

    slide.shapes.add_group_shape()

    left = top = width = height = Inches(1)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    p = tf.add_paragraph()
    p.text = "This is a first paragraph inside the group"

    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    p = tf.add_paragraph()
    p.text = "This is a second paragraph inside the group"

    pptx_stream = BytesIO()
    prs.save(pptx_stream)
    return pptx_stream


@pytest.fixture
def pptx_stream_with_bullet(tmp_path):
    prs = Presentation()
    bullet_slide_layout = prs.slide_layouts[1]

    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes

    title_shape = shapes.title
    body_shape = shapes.placeholders[1]

    title_shape.text = "Adding a Bullet Slide"

    tf = body_shape.text_frame
    tf.text = "Find the bullet slide layout"

    p = tf.add_paragraph()
    p.text = "Use _TextFrame.text for first bullet"
    p.level = 1

    p = tf.add_paragraph()
    p.text = "Use _TextFrame.add_paragraph() for subsequent bullets"
    p.level = 2

    pptx_stream = BytesIO()
    prs.save(pptx_stream)

    return pptx_stream


@pytest.fixture
def pptx_stream_with_table(tmp_path):
    prs = Presentation()
    title_only_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(title_only_slide_layout)
    shapes = slide.shapes

    shapes.title.text = "Adding a Table"

    rows = cols = 2
    left = top = Inches(2.0)
    width = Inches(6.0)
    height = Inches(0.8)

    table = shapes.add_table(rows, cols, left, top, width, height).table

    # set column widths
    table.columns[0].width = Inches(2.0)
    table.columns[1].width = Inches(4.0)

    # write column headings
    table.cell(0, 0).text = "Foo"
    table.cell(0, 1).text = "Bar"

    # write body cells
    table.cell(1, 0).text = "Baz"
    table.cell(1, 1).text = "Qux"

    pptx_stream = BytesIO()
    prs.save(pptx_stream)

    return pptx_stream


@pytest.fixture
def pptx_stream_with_image(tmp_path):
    imarray = numpy.random.rand(100, 100, 3) * 255
    img = Image.fromarray(imarray.astype("uint8")).convert("RGBA")
    img_stream = BytesIO()
    img.save(img_stream, format="png")

    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)

    left = top = Inches(1)
    slide.shapes.add_picture(img_stream, left, top)

    left = top = Inches(2)
    width = height = Inches(1)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    p = tf.add_paragraph()
    p.text = "This is text"

    pptx_stream = BytesIO()
    prs.save(pptx_stream)

    return pptx_stream


def test_pptx(pptx_stream_with_text, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_text,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2

    for data in extracted_data:
        assert len(data) == 3
        assert data[0] == "text"
        assert data[1]["source_metadata"]["source_id"] == "source1"
        assert data[1]["text_metadata"]["text_type"] == "page"
        assert isinstance(data[2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    Hello, World!
    =============

    This is a subtitle.
    -------------------
    """
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_0.rstrip()

    expected_1 = dedent(
        """\
    This is text inside a textbox

    **This is a second paragraph that's bold**

    *This is a third paragraph that's italic*

    <u>This is a fourth paragraph that's underlined</u>

    [link to NVIDIA](https://www.nvidia.com/en-us/)
    """
    )
    assert extracted_data[1][1]["content"].rstrip() == expected_1.rstrip()


def test_pptx_with_multiple_runs_in_title(pptx_stream_with_multiple_runs_in_title, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_multiple_runs_in_title,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1

    for data in extracted_data:
        assert len(data) == 3
        assert data[0] == "text"
        assert data[1]["source_metadata"]["source_id"] == "source1"
        assert data[1]["text_metadata"]["text_type"] == "page"
        assert isinstance(data[2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    Hello, World!
    =============

    Subtitle here.
    --------------
    """
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_0.rstrip()


def test_pptx_text_depth_presentation(pptx_stream_with_text, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_text,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="document",
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0] == "text"
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert extracted_data[0][1]["text_metadata"]["text_type"] == "document"
    assert isinstance(extracted_data[0][2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    Hello, World!
    =============

    This is a subtitle.
    -------------------

    This is text inside a textbox

    **This is a second paragraph that's bold**

    *This is a third paragraph that's italic*

    <u>This is a fourth paragraph that's underlined</u>

    [link to NVIDIA](https://www.nvidia.com/en-us/)
        """
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_0.rstrip()

    # Make sure there are no execssive newline characters. (Max is two characters \n\n.)
    assert "\n\n\n" not in extracted_data[0][1]["content"]


def test_pptx_text_depth_shape(pptx_stream_with_text, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_text,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth="block",
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 3

    for data in extracted_data:
        assert len(data) == 3
        assert data[0] == "text"
        assert data[1]["source_metadata"]["source_id"] == "source1"
        assert data[1]["text_metadata"]["text_type"] == "block"
        assert isinstance(data[2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    Hello, World!
    =============
        """
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_0.rstrip()

    expected_1 = dedent(
        """\
    This is a subtitle.
    -------------------
        """
    )
    assert extracted_data[1][1]["content"].rstrip() == expected_1.rstrip()

    expected_2 = dedent(
        """\
    This is text inside a textbox

    **This is a second paragraph that's bold**

    *This is a third paragraph that's italic*

    <u>This is a fourth paragraph that's underlined</u>

    [link to NVIDIA](https://www.nvidia.com/en-us/)
        """
    )
    assert extracted_data[2][1]["content"].rstrip() == expected_2.rstrip()


@pytest.mark.parametrize("text_depth", ["line", "span"])
def test_pptx_text_depth_para_run(pptx_stream_with_text, document_df, text_depth):
    extracted_data = python_pptx(
        pptx_stream_with_text,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
        text_depth=text_depth,
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 7

    for data in extracted_data:
        assert len(data) == 3
        assert data[0] == "text"
        assert data[1]["source_metadata"]["source_id"] == "source1"
        assert data[1]["text_metadata"]["text_type"] == text_depth
        assert isinstance(data[2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    Hello, World!
    =============

    This is a subtitle.
    -------------------

    This is text inside a textbox

    **This is a second paragraph that's bold**

    *This is a third paragraph that's italic*

    <u>This is a fourth paragraph that's underlined</u>

    [link to NVIDIA](https://www.nvidia.com/en-us/)
    """
    ).split("\n\n")
    for extracted, expected in zip(extracted_data, expected_0):
        assert extracted[1]["content"].rstrip() == expected.rstrip()


def test_pptx_bullet(pptx_stream_with_bullet, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_bullet,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0] == "text"
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert isinstance(extracted_data[0][2], str)

    # validate parsed text
    expected_content = dedent(
        """\
    Adding a Bullet Slide
    =====================

    * Find the bullet slide layout

      * Use \_TextFrame\.text for first bullet

        * Use \_TextFrame\.add\_paragraph\(\) for subsequent bullets
        """  # noqa: W605
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_content.rstrip()


def test_pptx_group(pptx_stream_with_group, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_group,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1

    assert len(extracted_data[0]) == 3
    assert extracted_data[0][0] == "text"
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert extracted_data[0][1]["text_metadata"]["text_type"] == "page"
    assert isinstance(extracted_data[0][2], str)

    # validate parsed text
    expected_0 = dedent(
        """\
    This is a first paragraph inside the group

    This is a second paragraph inside the group
    """
    )
    assert extracted_data[0][1]["content"].rstrip() == expected_0.rstrip()


def test_pptx_table(pptx_stream_with_table, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_table,
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3
    assert len(extracted_data[1]) == 3
    assert extracted_data[0][0] == "structured"
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert isinstance(extracted_data[0][2], str)
    assert extracted_data[1][0] == "text"
    assert extracted_data[1][1]["source_metadata"]["source_id"] == "source1"
    assert isinstance(extracted_data[1][2], str)

    # validate parsed text
    expected_content = dedent(
        """\
    | Foo   | Bar   |
    |:------|:------|
    | Baz   | Qux   |
        """
    )
    assert extracted_data[0][1]["table_metadata"]["table_content"].rstrip() == expected_content.rstrip()


def test_pptx_image(pptx_stream_with_image, document_df):
    extracted_data = python_pptx(
        pptx_stream_with_image,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        row_data=document_df.iloc[0],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 2
    assert len(extracted_data[0]) == 3

    assert extracted_data[0][0] == "text"
    assert extracted_data[0][1]["source_metadata"]["source_id"] == "source1"
    assert isinstance(extracted_data[0][2], str)

    assert extracted_data[1][0] == "image"
    assert extracted_data[1][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
