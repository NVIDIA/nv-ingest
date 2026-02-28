# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import os

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, ANY

import nv_ingest_api.internal.extract.pdf.engines.pdfium as module_under_test
from nv_ingest_api.internal.extract.pdf.engines.pdfium import (
    _extract_page_elements_using_image_ensemble,
)
from nv_ingest_api.util.metadata.aggregators import CroppedImageWithContent
from api.api_tests.utilities_for_test import get_project_root

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


def is_base64(sb):
    try:
        if isinstance(sb, str):
            # Ensure it's padded correctly
            sb_bytes = sb.encode("utf-8")
        else:
            sb_bytes = sb
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False


@pytest.fixture
def dummy_pages():
    return [
        (0, np.zeros((100, 100, 3), dtype=np.uint8), (10, 10)),
        (1, np.ones((100, 100, 3), dtype=np.uint8), (5, 5)),
    ]


@pytest.fixture
def dummy_pdf_stream():
    return io.BytesIO(b"dummy_pdf_data")


@pytest.fixture
def dummy_extractor_config():
    return {
        "row_data": {
            "source_id": "abc123",
            "metadata": {
                "source_metadata": {
                    "source_location": "s3://bucket/file.pdf",
                    "collection_id": "col123",
                    "partition_id": 1,
                    "access_level": -1,
                }
            },
        },
        "text_depth": "page",
        "table_output_format": "markdown",
        "extract_images_method": "simple",
        "extract_images_params": {},
        "pdfium_config": {
            "workers_per_progress_engine": 2,
            "yolox_endpoints": ("grpc", "http"),
        },
    }


@pytest.fixture
def pdf_stream_test_page_form_pdf():
    data_path = os.path.join(get_project_root(__file__), "data", "test-page-form.pdf")
    with open(data_path, "rb") as f:
        pdf_stream = io.BytesIO(f.read())
    return pdf_stream


@pytest.fixture
def pdf_stream_test_shapes_pdf():
    data_path = os.path.join(get_project_root(__file__), "data", "test-shapes.pdf")
    with open(data_path, "rb") as f:
        pdf_stream = io.BytesIO(f.read())
    return pdf_stream


@pytest.mark.xfail(reason="TODO: Fix this test")
def test_extract_page_elements_happy_path(dummy_pages):
    # Mock inference result to return dummy annotations for each page
    dummy_inference_results = [
        {"dummy_annotation": "page0"},
        {"dummy_annotation": "page1"},
    ]

    mock_yolox_client = MagicMock()
    mock_yolox_client.infer.return_value = dummy_inference_results

    # Patch _extract_page_element_images and monitor calls
    with patch(f"{MODULE_UNDER_TEST}._extract_page_element_images") as mock_extract_images:
        # Mock _extract_page_element_images to append dummy values to the page_elements list
        def fake_extract(annotation_dict, image, page_index, page_elements, padding_offset):
            page_elements.append((page_index, annotation_dict))

        mock_extract_images.side_effect = fake_extract

        # Act
        result = _extract_page_elements_using_image_ensemble(
            dummy_pages,
            yolox_client=mock_yolox_client,
            execution_trace_log=["dummy_trace"],
        )

        # Assert yolox_client called once
        mock_yolox_client.infer.assert_called_once_with(
            {"images": [dummy_pages[0][1], dummy_pages[1][1]]},
            max_batch_size=module_under_test.YOLOX_MAX_BATCH_SIZE,
            trace_info=["dummy_trace"],
            stage_name="pdf_content_extractor",
        )

        # Assert _extract_page_element_images called twice, once per page
        assert mock_extract_images.call_count == 2

        # Validate result contains expected data from fake_extract
        assert result == [
            (0, {"dummy_annotation": "page0"}),
            (1, {"dummy_annotation": "page1"}),
        ]


def test_extract_page_elements_handles_timeout(dummy_pages):
    mock_yolox_client = MagicMock()
    mock_yolox_client.infer.side_effect = TimeoutError("Timeout!")

    with patch(f"{MODULE_UNDER_TEST}._extract_page_element_images") as mock_extract_images:
        with pytest.raises(TimeoutError):
            _extract_page_elements_using_image_ensemble(
                dummy_pages,
                yolox_client=mock_yolox_client,
            )
        # Ensure _extract_page_element_images was never called
        mock_extract_images.assert_not_called()


def test_extract_page_elements_handles_generic_error(dummy_pages):
    mock_yolox_client = MagicMock()
    mock_yolox_client.infer.side_effect = RuntimeError("Some generic error")

    with patch(f"{MODULE_UNDER_TEST}._extract_page_element_images") as mock_extract_images:
        with pytest.raises(RuntimeError, match="Some generic error"):
            _extract_page_elements_using_image_ensemble(
                dummy_pages,
                yolox_client=mock_yolox_client,
            )
        mock_extract_images.assert_not_called()


def test_extract_page_element_images_happy_path_real_ops():
    annotation_dict = {
        "table": [[10, 20, 100, 150, 0.9]],
        "chart": [],
        "infographic": [],
    }
    dummy_image = np.ones((200, 200, 3), dtype=np.uint8) * 255

    page_elements = []

    module_under_test._extract_page_element_images(
        annotation_dict,
        dummy_image,
        page_idx=0,
        page_elements=page_elements,
        padding_offset=(5, 5),
    )

    assert len(page_elements) == 1
    page_idx, element = page_elements[0]
    assert page_idx == 0
    assert isinstance(element, CroppedImageWithContent)
    assert element.bbox == (5, 15, 95, 145)
    assert element.max_width == 190
    assert element.max_height == 190
    assert element.type_string == "table"
    assert isinstance(element.image, str)
    assert element.image.strip() != ""

    # ✅ Validate that it's a decodable base64 string
    try:
        _ = base64.b64decode(element.image, validate=True)
    except Exception:
        pytest.fail("Extracted image is not a valid base64 string")


def test_extract_page_element_images_ignores_missing_label():
    annotation_dict = {}  # No expected labels
    dummy_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    page_elements = []

    module_under_test._extract_page_element_images(
        annotation_dict,
        dummy_image,
        page_idx=1,
        page_elements=page_elements,
        padding_offset=(0, 0),
    )

    assert page_elements == []  # No elements should be extracted


def test_extract_page_element_images_ignores_missing_bbox():
    annotation_dict = {
        "table": [],
        "chart": [],
        "infographic": [],
    }
    dummy_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    page_elements = []

    module_under_test._extract_page_element_images(
        annotation_dict,
        dummy_image,
        page_idx=2,
        page_elements=page_elements,
    )

    assert page_elements == []  # Should gracefully do nothing


def test_extract_page_element_images_multiple_labels(dummy_image=np.ones((200, 200, 3), dtype=np.uint8) * 255):
    annotation_dict = {
        "chart": [[30, 40, 80, 100, 0.8]],
        "table": [[10, 20, 50, 60, 0.9]],
    }
    page_elements = []

    module_under_test._extract_page_element_images(
        annotation_dict,
        dummy_image,
        page_idx=3,
        page_elements=page_elements,
        padding_offset=(0, 0),
    )

    assert len(page_elements) == 2
    types = [element.type_string for _, element in page_elements]
    assert "table" in types
    assert "chart" in types

    # Validate that each image is a valid base64 string
    for _, element in page_elements:
        assert isinstance(element.image, str)
        decoded = base64.b64decode(element.image)
        assert decoded and isinstance(decoded, bytes)


def test_extract_page_text_happy_path():
    # Mock the text page and its method
    mock_textpage = MagicMock()
    mock_textpage.get_text_bounded.return_value = "Sample extracted text"

    # Mock the page to return the text page
    mock_page = MagicMock()
    mock_page.get_textpage.return_value = mock_textpage

    # Call the function
    result = module_under_test._extract_page_text(mock_page)

    # Validate the correct calls and result
    mock_page.get_textpage.assert_called_once()
    mock_textpage.get_text_bounded.assert_called_once()
    assert result == "Sample extracted text"


@pytest.mark.parametrize("method", ["simple", "group"])
@patch(f"{MODULE_UNDER_TEST}.construct_image_metadata_from_pdf_image")
@patch(f"{MODULE_UNDER_TEST}.extract_image_like_objects_from_pdfium_page")
@patch(f"{MODULE_UNDER_TEST}.extract_nested_simple_images_from_pdfium_page")
def test_extract_page_images_happy_path(
    mock_extract_simple,
    mock_extract_group,
    mock_construct_image_metadata,
    method,
):
    dummy_image_data = [MagicMock(), MagicMock()]

    # Setup mocks
    if method == "simple":
        mock_extract_simple.return_value = dummy_image_data
    else:
        mock_extract_group.return_value = dummy_image_data

    mock_construct_image_metadata.side_effect = ["image_meta_1", "image_meta_2"]

    # Call function
    result = module_under_test._extract_page_images(
        extract_images_method=method,
        page=MagicMock(),
        page_idx=0,
        page_width=100,
        page_height=200,
        page_count=5,
        source_metadata={"source_id": "abc"},
        base_unified_metadata={"doc_id": "doc1"},
        dummy_param="dummy_value",  # extra params should be accepted by the group method
    )

    # Assertions
    if method == "simple":
        mock_extract_simple.assert_called_once_with(ANY)
        mock_extract_group.assert_not_called()
    else:
        mock_extract_group.assert_called_once_with(ANY, merge=True, dummy_param="dummy_value")
        mock_extract_simple.assert_not_called()

    assert mock_construct_image_metadata.call_count == 2
    assert result == ["image_meta_1", "image_meta_2"]


@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
def test_extract_page_elements_happy_path2(
    mock_create_client,
    mock_extract_ensemble,
    mock_construct_metadata,
):
    # Setup dummy returns
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_element = MagicMock()
    mock_element.type_string = "table"
    mock_element_results = [(0, mock_element), (1, mock_element)]
    mock_extract_ensemble.return_value = mock_element_results
    mock_construct_metadata.side_effect = ["meta_0", "meta_1"]

    dummy_pages = [MagicMock()]
    dummy_source_metadata = {"source": "dummy"}
    dummy_base_metadata = {"doc": "dummy"}
    dummy_trace_log = []

    result = module_under_test._extract_page_elements(
        pages=dummy_pages,
        page_count=2,
        source_metadata=dummy_source_metadata,
        base_unified_metadata=dummy_base_metadata,
        extract_tables=True,
        extract_charts=False,
        extract_infographics=False,
        page_to_text_flag_map={0: False},
        table_output_format="markdown",
        yolox_endpoints=("grpc://dummy", "http://dummy"),
        yolox_infer_protocol="http",
        auth_token="dummy_token",
        execution_trace_log=dummy_trace_log,
    )

    # Assertions
    mock_create_client.assert_called_once()
    mock_extract_ensemble.assert_called_once_with(dummy_pages, mock_client, execution_trace_log=dummy_trace_log)
    assert mock_construct_metadata.call_count == 2
    assert result == ["meta_0", "meta_1"]


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
def test_extract_page_elements_fallback_to_default_model(
    mock_construct_metadata,
    mock_extract_ensemble,
    mock_create_client,
):
    # Fallback to 'yolox' if model name fetch fails
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_element = MagicMock()
    mock_element.type_string = "chart"
    mock_extract_ensemble.return_value = [(0, mock_element)]
    mock_construct_metadata.return_value = "chart_meta"

    result = module_under_test._extract_page_elements(
        pages=[MagicMock()],
        page_count=1,
        source_metadata={},
        base_unified_metadata={},
        extract_tables=False,
        extract_charts=True,
        extract_infographics=False,
        page_to_text_flag_map={0: False},
        table_output_format="latex",
        yolox_endpoints=("grpc://dummy", "http://dummy"),
        yolox_infer_protocol="http",
    )

    mock_create_client.assert_called_once()
    mock_extract_ensemble.assert_called_once()
    mock_construct_metadata.assert_called_once()
    assert result == ["chart_meta"]


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
def test_extract_page_elements_filters_by_flags(mock_extract_ensemble, mock_create_client):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # Element with type 'infographic'
    mock_element = MagicMock()
    mock_element.type_string = "infographic"
    mock_extract_ensemble.return_value = [(0, mock_element)]

    result = module_under_test._extract_page_elements(
        pages=[MagicMock()],
        page_count=1,
        source_metadata={},
        base_unified_metadata={},
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,  # all False => should be skipped
        page_to_text_flag_map={0: False},
        table_output_format="latex",
        yolox_endpoints=("grpc://dummy", None),
        yolox_infer_protocol="http",
    )

    # Should skip everything, hence no metadata constructed, result empty
    mock_extract_ensemble.assert_called_once()
    assert result == []


@patch(f"{MODULE_UNDER_TEST}.construct_image_metadata_from_base64")
@patch(f"{MODULE_UNDER_TEST}.numpy_to_base64")
@patch(f"{MODULE_UNDER_TEST}.construct_text_metadata")
@patch(f"{MODULE_UNDER_TEST}._extract_page_images")
@patch(f"{MODULE_UNDER_TEST}.pdfium_pages_to_numpy")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements")
@patch(f"{MODULE_UNDER_TEST}.extract_pdf_metadata")
@patch(f"{MODULE_UNDER_TEST}.libpdfium.PdfDocument")
def test_pdfium_extractor_happy_path(
    mock_pdf_doc,
    mock_extract_metadata,
    mock_extract_elements,
    mock_pages_to_numpy,
    mock_extract_images,
    mock_construct_text_metadata,
    mock_numpy_to_base64,
    mock_construct_image_metadata_from_base64,
    dummy_pdf_stream,
    dummy_extractor_config,
):
    # Setup mock PDF
    mock_doc = MagicMock()
    mock_pdf_doc.return_value = mock_doc
    mock_page = MagicMock()
    mock_page.get_size.return_value = (1000, 1000)
    mock_doc.get_page.return_value = mock_page
    mock_doc.__len__.return_value = 1

    # PDF metadata
    mock_pdf_metadata = MagicMock()
    mock_pdf_metadata.page_count = 1
    mock_pdf_metadata.filename = "file.pdf"
    mock_pdf_metadata.source_type = "pdf"
    mock_pdf_metadata.keywords = ["test"]
    mock_pdf_metadata.date_created = "2024-01-01"
    mock_pdf_metadata.last_modified = "2024-01-02"
    mock_extract_metadata.return_value = mock_pdf_metadata

    # Mock return values
    mock_construct_text_metadata.return_value = "text_meta"
    mock_extract_images.return_value = ["image_meta"]
    mock_pages_to_numpy.return_value = ([MagicMock()], [MagicMock()])
    mock_extract_elements.return_value = ["table_chart_meta"]
    mock_numpy_to_base64.return_value = "image_base64"
    mock_construct_image_metadata_from_base64.return_value = "page_image_meta"

    # Call
    result = module_under_test.pdfium_extractor(
        dummy_pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_infographics=True,
        extract_tables=True,
        extract_charts=True,
        extract_page_as_image=True,
        extractor_config=dummy_extractor_config,
        execution_trace_log=[],
    )

    # Validations
    mock_pdf_doc.assert_called_once_with(dummy_pdf_stream)
    mock_extract_metadata.assert_called_once()
    mock_construct_text_metadata.assert_called()
    mock_extract_images.assert_called()
    mock_pages_to_numpy.assert_called()
    mock_extract_elements.assert_called()
    assert result == ["text_meta", "image_meta", "page_image_meta", "table_chart_meta"]


@patch(f"{MODULE_UNDER_TEST}.extract_pdf_metadata")
@patch(f"{MODULE_UNDER_TEST}.libpdfium.PdfDocument")
def test_pdfium_extractor_invalid_config_raises(mock_pdf_doc, mock_extract_metadata):
    with pytest.raises(
        ValueError,
        match="`extractor_config` must include a valid 'row_data' dictionary.",
    ):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config={},
        )

    with pytest.raises(ValueError, match="The 'row_data' dictionary must contain the 'source_id' key."):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config={"row_data": {}},
        )

    with pytest.raises(ValueError, match="Invalid text_depth: invalid_depth"):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config={
                "row_data": {"source_id": "abc"},
                "text_depth": "invalid_depth",
            },
        )

    with pytest.raises(ValueError, match="Invalid table_output_format: invalid_format"):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config={
                "row_data": {"source_id": "abc"},
                "table_output_format": "invalid_format",
            },
        )


@patch(f"{MODULE_UNDER_TEST}.extract_pdf_metadata")
@patch(f"{MODULE_UNDER_TEST}.libpdfium.PdfDocument")
def test_pdfium_extractor_invalid_pdfium_config_raises(mock_pdf_doc, mock_extract_metadata):
    with pytest.raises(
        ValueError,
        match="`pdfium_config` must be a dictionary or a PDFiumConfigSchema instance.",
    ):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            extract_text=True,
            extract_images=True,
            extract_infographics=True,
            extract_tables=True,
            extract_charts=True,
            extract_page_as_image=True,
            extractor_config={
                "row_data": {"source_id": "abc"},
                "pdfium_config": "not_a_dict",
            },
        )


def test_pdfium_extractor_page_form(pdf_stream_test_page_form_pdf, dummy_extractor_config):
    extractor_config = dummy_extractor_config.copy()
    extractor_config["extract_images_method"] = "simple"

    extracted_data = module_under_test.pdfium_extractor(
        pdf_stream_test_page_form_pdf,
        extract_text=False,
        extract_images=True,
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extract_page_as_image=False,
        extractor_config=extractor_config,
        execution_trace_log=[],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 18
    assert all(data[0].value == "image" for data in extracted_data)
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][0]) == 595 for data in extracted_data)
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][1]) == 841 for data in extracted_data)

    extractor_config = dummy_extractor_config.copy()
    extractor_config["extract_images_method"] = "group"

    extracted_data = module_under_test.pdfium_extractor(
        pdf_stream_test_page_form_pdf,
        extract_text=False,
        extract_images=True,
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extract_page_as_image=False,
        extractor_config=extractor_config,
        execution_trace_log=[],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 1
    assert extracted_data[0][0].value == "image"
    assert extracted_data[0][1]["image_metadata"]["image_location"] == (
        306,
        219,
        524,
        375,
    )
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][0]) == 595 for data in extracted_data)
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][1]) == 841 for data in extracted_data)


def test_pdfium_extractor_shapes_grouped(pdf_stream_test_shapes_pdf, dummy_extractor_config):
    extractor_config = dummy_extractor_config.copy()
    extractor_config["extract_images_method"] = "group"

    extracted_data = module_under_test.pdfium_extractor(
        pdf_stream_test_shapes_pdf,
        extract_text=False,
        extract_images=True,
        extract_infographics=False,
        extract_tables=False,
        extract_charts=False,
        extract_page_as_image=False,
        extractor_config=extractor_config,
        execution_trace_log=[],
    )

    assert isinstance(extracted_data, list)
    assert len(extracted_data) == 3
    assert all(data[0].value == "image" for data in extracted_data)
    assert extracted_data[0][1]["image_metadata"]["image_location"] == (
        149,
        33,
        549,
        308,
    )
    assert extracted_data[1][1]["image_metadata"]["image_location"] == (
        36,
        56,
        131,
        114,
    )
    assert extracted_data[2][1]["image_metadata"]["image_location"] == (538, 0, 561, 33)
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][0]) == 595 for data in extracted_data)
    assert all(int(data[1]["image_metadata"]["image_location_max_dimensions"][1]) == 419 for data in extracted_data)


@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
def test_extract_page_elements_processes_text_when_flag_is_true(
    mock_create_client,
    mock_extract_ensemble,
    mock_construct_metadata,
):
    """
    Verify that text elements ARE processed when the flag in page_to_text_flag_map is True.
    """
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # Simulate YOLOX detecting a 'paragraph'
    mock_element = MagicMock()
    mock_element.type_string = "paragraph"
    mock_extract_ensemble.return_value = [(0, mock_element)]  # Page index 0

    module_under_test._extract_page_elements(
        pages=[MagicMock()],
        page_count=1,
        source_metadata={},
        base_unified_metadata={},
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        page_to_text_flag_map={0: True},  # <<< Key part of the test: Enable text processing for page 0
        table_output_format="markdown",
        yolox_endpoints=("grpc://dummy", None),
    )

    # Assert that since the flag was True, we attempted to construct metadata for the text element.
    mock_construct_metadata.assert_called_once()


@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
def test_extract_page_elements_skips_text_when_flag_is_false(
    mock_create_client,
    mock_extract_ensemble,
    mock_construct_metadata,
):
    """
    Verify that text elements ARE SKIPPED when the flag in page_to_text_flag_map is False.
    """
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # Simulate YOLOX detecting a 'paragraph'
    mock_element = MagicMock()
    mock_element.type_string = "paragraph"
    mock_extract_ensemble.return_value = [(0, mock_element)]  # Page index 0

    module_under_test._extract_page_elements(
        pages=[MagicMock()],
        page_count=1,
        source_metadata={},
        base_unified_metadata={},
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        page_to_text_flag_map={0: False},  # <<< Key part of the test: Disable text processing for page 0
        table_output_format="markdown",
        yolox_endpoints=("grpc://dummy", None),
    )

    # Assert that since the flag was False, we did NOT attempt to construct metadata.
    mock_construct_metadata.assert_not_called()


@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
def test_extract_page_elements_mixed_flags_and_types(
    mock_create_client,
    mock_extract_ensemble,
    mock_construct_metadata,
):
    """
    Verify correct behavior with a mix of element types and flags.
    """
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # Simulate YOLOX detecting a table on page 0 and a paragraph on page 1
    mock_table_element = MagicMock()
    mock_table_element.type_string = "table"
    mock_paragraph_element = MagicMock()
    mock_paragraph_element.type_string = "paragraph"

    mock_extract_ensemble.return_value = [(0, mock_table_element), (1, mock_paragraph_element)]

    module_under_test._extract_page_elements(
        pages=[MagicMock(), MagicMock()],
        page_count=2,
        source_metadata={},
        base_unified_metadata={},
        extract_tables=True,  # Enable table processing
        extract_charts=False,
        extract_infographics=False,
        # Enable text for page 1, but disable for page 0
        page_to_text_flag_map={0: False, 1: True},
        table_output_format="markdown",
        yolox_endpoints=("grpc://dummy", None),
    )

    # We expect construct_metadata to be called twice:
    # 1. For the table on page 0 (since extract_tables=True)
    # 2. For the paragraph on page 1 (since page_to_text_flag_map[1]=True)
    assert mock_construct_metadata.call_count == 2

    # Check the calls were made with the correct elements
    call_args_list = mock_construct_metadata.call_args_list
    # The order is not guaranteed, so we check the contents
    called_with_table = any(call.args[0] is mock_table_element for call in call_args_list)
    called_with_paragraph = any(call.args[0] is mock_paragraph_element for call in call_args_list)

    assert called_with_table, "construct_page_element_metadata was not called for the table element"
    assert called_with_paragraph, "construct_page_element_metadata was not called for the paragraph element"
