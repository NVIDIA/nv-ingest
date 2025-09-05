# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, ANY

import nv_ingest_api.internal.extract.pdf.engines.pdfium as module_under_test
from nv_ingest_api.internal.extract.pdf.engines.pdfium import _extract_page_elements_using_image_ensemble
from nv_ingest_api.util.metadata.aggregators import CroppedImageWithContent

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
                    "access_level": "public",
                }
            },
        },
        "text_depth": "page",
        "paddle_output_format": "markdown",
        "extract_images_method": "simple",
        "extract_images_params": {},
        "pdfium_config": {"workers_per_progress_engine": 2, "yolox_endpoints": ("grpc", "http")},
    }


@pytest.mark.xfail(reason="TODO: Fix this test")
def test_extract_page_elements_happy_path(dummy_pages):
    # Mock inference result to return dummy annotations for each page
    dummy_inference_results = [
        {"dummy_annotation": "page0"},
        {"dummy_annotation": "page1"},
    ]

    mock_yolox_client = MagicMock()
    mock_yolox_client.infer.return_value = dummy_inference_results

    def _images_from_infer_call(args, kwargs):
        """Return the images list regardless of whether the payload was passed
        as a positional dict or as kwargs. Fail loudly if missing."""
        if args and isinstance(args[0], dict) and "images" in args[0]:
            return args[0]["images"]
        if "images" in kwargs:
            return kwargs["images"]
        pytest.fail("yolox_client.infer() was called without an 'images' payload")

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

    # Assert
    assert mock_yolox_client.infer.call_count == 1, "infer() should be called exactly once"
    args, kwargs = mock_yolox_client.infer.call_args

    # Focused kwarg checks (shape-agnostic)
    assert kwargs.get("model_name") == "yolox", "Unexpected model_name"
    assert kwargs.get("stage_name") == "pdf_extraction", "Unexpected stage_name"
    assert kwargs.get("trace_info") == ["dummy_trace"], "trace_info should carry through"

    if hasattr(MODULE_UNDER_TEST, "YOLOX_MAX_BATCH_SIZE"):
        assert kwargs.get("max_batch_size") == MODULE_UNDER_TEST.YOLOX_MAX_BATCH_SIZE, "Incorrect max_batch_size"
    else:
        assert "max_batch_size" in kwargs, "max_batch_size missing"

    # Images payload (robust to positional vs kwargs)
    imgs = _images_from_infer_call(args, kwargs)
    assert len(imgs) == 2, "Expected exactly two page images"
    assert imgs[0] is dummy_pages[0][1], "First image object mismatch"
    assert imgs[1] is dummy_pages[1][1], "Second image object mismatch"

    # Helper called once per page
    assert mock_extract_images.call_count == 2, "_extract_page_element_images should be called per page"

    # Result assembled by fake_extract
    assert result == [
        (0, {"dummy_annotation": "page0"}),
        (1, {"dummy_annotation": "page1"}),
    ], "Aggregated result mismatch"


def test_extract_page_elements_handles_timeout(dummy_pages):
    mock_yolox_client = MagicMock()
    mock_yolox_client.infer.side_effect = TimeoutError("Timeout!")

    with patch(f"{MODULE_UNDER_TEST}._extract_page_element_images") as mock_extract_images:
        with pytest.raises(TimeoutError):
            _extract_page_elements_using_image_ensemble(
                dummy_pages,
                yolox_client=mock_yolox_client,
                yolox_model_name="yolox",
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
@patch(f"{MODULE_UNDER_TEST}.get_yolox_model_name")
def test_extract_page_elements_happy_path2(
    mock_get_model_name,
    mock_create_client,
    mock_extract_ensemble,
    mock_construct_metadata,
):
    # Setup dummy returns
    mock_get_model_name.return_value = "custom_yolox"
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
        paddle_output_format="markdown",
        yolox_endpoints=("grpc://dummy", "http://dummy"),
        yolox_infer_protocol="http",
        auth_token="dummy_token",
        execution_trace_log=dummy_trace_log,
    )

    # Assertions
    mock_get_model_name.assert_called_once_with("http://dummy")
    mock_create_client.assert_called_once()
    mock_extract_ensemble.assert_called_once_with(
        dummy_pages, mock_client, "custom_yolox", execution_trace_log=dummy_trace_log
    )
    assert mock_construct_metadata.call_count == 2
    assert result == ["meta_0", "meta_1"]
    mock_client.close.assert_called_once()


@patch(f"{MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{MODULE_UNDER_TEST}.get_yolox_model_name", side_effect=Exception("model error"))
@patch(f"{MODULE_UNDER_TEST}._extract_page_elements_using_image_ensemble")
@patch(f"{MODULE_UNDER_TEST}.construct_page_element_metadata")
def test_extract_page_elements_fallback_to_default_model(
    mock_construct_metadata,
    mock_extract_ensemble,
    mock_get_model_name,
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
        paddle_output_format="latex",
        yolox_endpoints=("grpc://dummy", "http://dummy"),
        yolox_infer_protocol="http",
    )

    mock_create_client.assert_called_once()
    mock_extract_ensemble.assert_called_once()
    mock_construct_metadata.assert_called_once()
    assert result == ["chart_meta"]
    mock_client.close.assert_called_once()


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
        paddle_output_format="latex",
        yolox_endpoints=("grpc://dummy", None),
        yolox_infer_protocol="http",
    )

    # Should skip everything, hence no metadata constructed, result empty
    mock_extract_ensemble.assert_called_once()
    assert result == []
    mock_client.close.assert_called_once()


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

    # Call
    result = module_under_test.pdfium_extractor(
        dummy_pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_infographics=True,
        extract_tables=True,
        extract_charts=True,
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
    mock_doc.close.assert_called_once()
    assert result == ["text_meta", "image_meta", "table_chart_meta"]


@patch(f"{MODULE_UNDER_TEST}.extract_pdf_metadata")
@patch(f"{MODULE_UNDER_TEST}.libpdfium.PdfDocument")
def test_pdfium_extractor_invalid_config_raises(mock_pdf_doc, mock_extract_metadata):
    with pytest.raises(ValueError, match="`extractor_config` must include a valid 'row_data' dictionary."):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            True,
            True,
            True,
            True,
            True,
            extractor_config={},
        )

    with pytest.raises(ValueError, match="The 'row_data' dictionary must contain the 'source_id' key."):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            True,
            True,
            True,
            True,
            True,
            extractor_config={"row_data": {}},
        )

    with pytest.raises(ValueError, match="Invalid text_depth: invalid_depth"):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            True,
            True,
            True,
            True,
            True,
            extractor_config={"row_data": {"source_id": "abc"}, "text_depth": "invalid_depth"},
        )

    with pytest.raises(ValueError, match="Invalid paddle_output_format: invalid_format"):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            True,
            True,
            True,
            True,
            True,
            extractor_config={"row_data": {"source_id": "abc"}, "paddle_output_format": "invalid_format"},
        )


@patch(f"{MODULE_UNDER_TEST}.extract_pdf_metadata")
@patch(f"{MODULE_UNDER_TEST}.libpdfium.PdfDocument")
def test_pdfium_extractor_invalid_pdfium_config_raises(mock_pdf_doc, mock_extract_metadata):
    with pytest.raises(ValueError, match="`pdfium_config` must be a dictionary or a PDFiumConfigSchema instance."):
        module_under_test.pdfium_extractor(
            io.BytesIO(),
            True,
            True,
            True,
            True,
            True,
            extractor_config={"row_data": {"source_id": "abc"}, "pdfium_config": "not_a_dict"},
        )
