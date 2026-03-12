# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the render-at-target-scale optimisation in nemo_retriever's
``_render_page_to_base64`` and for the ``_is_scanned_page`` import switch,
and for the ``as_b64`` crop mode in ``_crop_all_from_page``.
"""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_extract = pytest.importorskip("nemo_retriever.pdf.extract")
_ocr = pytest.importorskip("nemo_retriever.ocr.ocr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_page(width: float = 612, height: float = 792, render_h: int = 100, render_w: int = 80):
    """Return a mock PdfPage (US Letter at 72 DPI points)."""
    page = MagicMock()
    page.get_width.return_value = width
    page.get_height.return_value = height

    bitmap = MagicMock()
    bitmap.to_numpy.return_value = np.zeros((render_h, render_w, 4), dtype=np.uint8)
    bitmap.mode = "BGRA"
    page.render.return_value = bitmap
    return page, bitmap


# ---------------------------------------------------------------------------
# Tests for _render_page_to_base64 — JPEG / PNG encoding
# ---------------------------------------------------------------------------


class TestRenderPageEncoding:
    """Verify JPEG and PNG encoding in _render_page_to_base64."""

    def test_renders_at_full_dpi(self):
        """page.render() should be called with full DPI scale."""
        page, bitmap = _make_mock_page()
        dpi = 200
        base_scale = dpi / 72.0

        _extract._render_page_to_base64(page, dpi=dpi, render_mode="full_dpi")

        render_call = page.render.call_args
        actual_scale = render_call.kwargs.get("scale", render_call.args[0] if render_call.args else None)
        assert abs(actual_scale - base_scale) < 0.001

    def test_jpeg_output(self):
        """Default image_format produces valid JPEG base64."""
        page, bitmap = _make_mock_page()
        result = _extract._render_page_to_base64(page, dpi=200, image_format="jpeg")

        assert result["encoding"] == "jpeg"
        raw = base64.b64decode(result["image_b64"])
        img = Image.open(io.BytesIO(raw))
        assert img.format == "JPEG"

    def test_png_output(self):
        """image_format='png' produces valid PNG base64."""
        page, bitmap = _make_mock_page()
        result = _extract._render_page_to_base64(page, dpi=200, image_format="png")

        assert result["encoding"] == "png"
        raw = base64.b64decode(result["image_b64"])
        img = Image.open(io.BytesIO(raw))
        assert img.format == "PNG"

    def test_orig_shape_hw(self):
        """orig_shape_hw should reflect the rendered raster size."""
        page, bitmap = _make_mock_page(render_h=200, render_w=150)
        result = _extract._render_page_to_base64(page, dpi=200)

        assert result["orig_shape_hw"] == (200, 150)


# ---------------------------------------------------------------------------
# Test _is_scanned_page import switch
# ---------------------------------------------------------------------------


class TestIsScannedPageImport:
    """Verify the re-exported _is_scanned_page works correctly."""

    def test_is_scanned_page_available(self):
        """The _is_scanned_page alias should be importable from extract."""
        assert hasattr(_extract, "_is_scanned_page")

    def test_is_scanned_page_matches_api(self):
        """The imported function should be the same object as the api/ version."""
        from nv_ingest_api.util.pdf.pdfium import is_scanned_page

        assert _extract._is_scanned_page is is_scanned_page


# ---------------------------------------------------------------------------
# Helpers for _crop_all_from_page tests
# ---------------------------------------------------------------------------


def _make_test_image_b64(width: int = 100, height: int = 80) -> str:
    """Create a base64-encoded PNG of a solid-colour RGB image."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_detections(*labels_and_bboxes):
    """Build a list of detection dicts from (label, bbox) pairs."""
    return [{"label_name": label, "bbox_xyxy_norm": list(bbox)} for label, bbox in labels_and_bboxes]


# ---------------------------------------------------------------------------
# Tests for _crop_all_from_page as_b64 mode
# ---------------------------------------------------------------------------


class TestCropAllFromPageAsB64:
    """Verify the as_b64 parameter on _crop_all_from_page."""

    def test_as_b64_false_returns_numpy(self):
        """Default mode returns numpy uint8 arrays."""
        img_b64 = _make_test_image_b64(100, 80)
        dets = _make_detections(("table", (0.1, 0.1, 0.9, 0.9)))
        results = _ocr._crop_all_from_page(img_b64, dets, {"table"}, as_b64=False)

        assert len(results) == 1
        label, bbox, value = results[0]
        assert label == "table"
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.uint8
        assert value.ndim == 3  # HWC

    def test_as_b64_true_returns_valid_base64_png(self):
        """as_b64=True returns a base64 string that decodes to a valid PNG."""
        img_b64 = _make_test_image_b64(100, 80)
        dets = _make_detections(("chart", (0.0, 0.0, 0.5, 0.5)))
        results = _ocr._crop_all_from_page(img_b64, dets, {"chart"}, as_b64=True)

        assert len(results) == 1
        label, bbox, value = results[0]
        assert label == "chart"
        assert isinstance(value, str)

        # Decode and verify it's a valid PNG image
        raw = base64.b64decode(value)
        img = Image.open(io.BytesIO(raw))
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_as_b64_default_is_numpy(self):
        """Calling without as_b64 kwarg preserves backward-compat numpy output."""
        img_b64 = _make_test_image_b64(100, 80)
        dets = _make_detections(("table", (0.1, 0.1, 0.9, 0.9)))
        results = _ocr._crop_all_from_page(img_b64, dets, {"table"})

        assert len(results) == 1
        _, _, value = results[0]
        assert isinstance(value, np.ndarray)

    def test_as_b64_skips_tiny_crops(self):
        """Crops with width or height <= 1 are skipped in both modes."""
        img_b64 = _make_test_image_b64(100, 80)
        # bbox that produces a 1-pixel-wide crop
        dets = _make_detections(("table", (0.0, 0.0, 0.01, 0.5)))
        results_np = _ocr._crop_all_from_page(img_b64, dets, {"table"}, as_b64=False)
        results_b64 = _ocr._crop_all_from_page(img_b64, dets, {"table"}, as_b64=True)
        assert len(results_np) == len(results_b64)

    def test_as_b64_multiple_detections(self):
        """Multiple valid detections all produce results in both modes."""
        img_b64 = _make_test_image_b64(200, 200)
        dets = _make_detections(
            ("table", (0.0, 0.0, 0.5, 0.5)),
            ("chart", (0.5, 0.5, 1.0, 1.0)),
        )
        results = _ocr._crop_all_from_page(img_b64, dets, {"table", "chart"}, as_b64=True)
        assert len(results) == 2
        assert results[0][0] == "table"
        assert results[1][0] == "chart"
        for _, _, value in results:
            assert isinstance(value, str)
            # Each must be valid base64
            base64.b64decode(value)


class TestOcrRemotePathNoNpRoundtrip:
    """Verify the remote path in ocr_page_elements doesn't call _np_rgb_to_b64_png."""

    def test_remote_path_skips_np_rgb_to_b64_png(self):
        """When invoke_url is set, _np_rgb_to_b64_png should never be called."""
        import pandas as pd

        img_b64 = _make_test_image_b64(100, 80)
        dets = _make_detections(("table", (0.1, 0.1, 0.9, 0.9)))
        df = pd.DataFrame(
            {
                "page_elements_v3": [{"detections": dets}],
                "page_image": [{"image_b64": img_b64}],
            }
        )

        with patch.object(_ocr, "_np_rgb_to_b64_png", wraps=_ocr._np_rgb_to_b64_png) as spy, patch.object(
            _ocr, "invoke_image_inference_batches", return_value=[{"text": "hello"}]
        ):
            _ocr.ocr_page_elements(
                df,
                invoke_url="http://fake-nim:8000/v1/ocr",
                extract_tables=True,
            )
            spy.assert_not_called()
