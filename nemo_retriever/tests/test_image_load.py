# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_retriever.image: load.py and ray_data.py."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import patch

import pandas as pd
import pytest

PIL = pytest.importorskip("PIL", reason="Pillow is required for image tests")
from PIL import Image  # noqa: E402

from nemo_retriever.image.load import (  # noqa: E402
    SUPPORTED_IMAGE_EXTENSIONS,
    image_bytes_to_pages_df,
    image_file_to_pages_df,
)
from nemo_retriever.image.ray_data import ImageLoadActor  # noqa: E402

# -- Helpers ------------------------------------------------------------------

_PAGE_SCHEMA_COLUMNS = {
    "path",
    "page_number",
    "source_id",
    "text",
    "page_image",
    "images",
    "tables",
    "charts",
    "infographics",
    "metadata",
}


def _make_image(width: int = 10, height: int = 10, color: str = "red") -> Image.Image:
    """Create a simple solid-color test image."""
    return Image.new("RGB", (width, height), color=color)


def _image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _assert_valid_page_df(df: pd.DataFrame, path: str, *, expect_error: bool = False) -> None:
    """Assert the DataFrame conforms to the PDF extraction page schema."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert set(df.columns) == _PAGE_SCHEMA_COLUMNS

    row = df.iloc[0]
    assert row["path"] == path
    assert row["page_number"] == 1
    assert row["source_id"] is None
    assert row["text"] == ""
    assert isinstance(row["images"], list)
    assert isinstance(row["tables"], list)
    assert isinstance(row["charts"], list)
    assert isinstance(row["infographics"], list)

    meta = row["metadata"]
    assert isinstance(meta, dict)
    assert meta["source_path"] == path

    if expect_error:
        assert meta["error"] is not None
        assert row["page_image"] is None
    else:
        assert meta["error"] is None
        assert meta["has_text"] is False
        assert meta["needs_ocr_for_text"] is True
        assert isinstance(meta["dpi"], int)

        pi = row["page_image"]
        assert isinstance(pi, dict)
        assert "image_b64" in pi
        assert pi["encoding"] == "png"
        assert isinstance(pi["orig_shape_hw"], tuple)
        assert len(pi["orig_shape_hw"]) == 2

        # Verify the base64 decodes to a valid PNG.
        decoded = base64.b64decode(pi["image_b64"])
        img = Image.open(BytesIO(decoded))
        assert img.format == "PNG"


# -- Tests: image_bytes_to_pages_df -------------------------------------------


class TestImageBytesToPagesDf:
    def test_png(self) -> None:
        img = _make_image()
        data = _image_to_bytes(img, "PNG")
        df = image_bytes_to_pages_df(data, "/tmp/test.png")
        _assert_valid_page_df(df, "/tmp/test.png")
        assert df.iloc[0]["page_image"]["orig_shape_hw"] == (10, 10)

    def test_jpeg(self) -> None:
        img = _make_image(20, 15)
        data = _image_to_bytes(img, "JPEG")
        df = image_bytes_to_pages_df(data, "/tmp/photo.jpg")
        _assert_valid_page_df(df, "/tmp/photo.jpg")
        assert df.iloc[0]["page_image"]["orig_shape_hw"] == (15, 20)

    def test_bmp(self) -> None:
        img = _make_image(8, 8)
        data = _image_to_bytes(img, "BMP")
        df = image_bytes_to_pages_df(data, "/data/scan.bmp")
        _assert_valid_page_df(df, "/data/scan.bmp")

    def test_tiff(self) -> None:
        img = _make_image(12, 12)
        data = _image_to_bytes(img, "TIFF")
        df = image_bytes_to_pages_df(data, "/data/doc.tiff")
        _assert_valid_page_df(df, "/data/doc.tiff")

    def test_tif_extension(self) -> None:
        img = _make_image(12, 12)
        data = _image_to_bytes(img, "TIFF")
        df = image_bytes_to_pages_df(data, "/data/doc.tif")
        _assert_valid_page_df(df, "/data/doc.tif")

    def test_corrupt_file_error_handling(self) -> None:
        df = image_bytes_to_pages_df(b"not-an-image", "/tmp/corrupt.png")
        _assert_valid_page_df(df, "/tmp/corrupt.png", expect_error=True)

    def test_unsupported_extension(self) -> None:
        df = image_bytes_to_pages_df(b"data", "/tmp/file.xyz")
        _assert_valid_page_df(df, "/tmp/file.xyz", expect_error=True)
        assert "Unsupported image format" in df.iloc[0]["metadata"]["error"]

    def test_svg_without_cairosvg(self) -> None:
        with patch.dict("sys.modules", {"cairosvg": None}):
            with pytest.raises(ImportError, match="cairosvg"):
                image_bytes_to_pages_df(b"<svg></svg>", "/tmp/icon.svg")


# -- Tests: image_file_to_pages_df -------------------------------------------


class TestImageFileToPagesDf:
    def test_roundtrip(self, tmp_path) -> None:
        img = _make_image(16, 16, "blue")
        p = tmp_path / "test.png"
        img.save(str(p), format="PNG")

        df = image_file_to_pages_df(str(p))
        _assert_valid_page_df(df, str(p.resolve()))
        assert df.iloc[0]["page_image"]["orig_shape_hw"] == (16, 16)

    def test_missing_file(self) -> None:
        df = image_file_to_pages_df("/nonexistent/missing.png")
        _assert_valid_page_df(df, "/nonexistent/missing.png", expect_error=True)


# -- Tests: ImageLoadActor ---------------------------------------------------


class TestImageLoadActor:
    def test_batch_processing(self) -> None:
        actor = ImageLoadActor()
        img1 = _image_to_bytes(_make_image(10, 10))
        img2 = _image_to_bytes(_make_image(20, 20))
        batch = pd.DataFrame(
            [
                {"bytes": img1, "path": "/a/img1.png"},
                {"bytes": img2, "path": "/b/img2.png"},
            ]
        )
        result = actor(batch)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["path"]) == ["/a/img1.png", "/b/img2.png"]
        assert result.iloc[0]["page_image"]["orig_shape_hw"] == (10, 10)
        assert result.iloc[1]["page_image"]["orig_shape_hw"] == (20, 20)

    def test_empty_batch(self) -> None:
        actor = ImageLoadActor()
        result = actor(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_columns_skipped(self) -> None:
        actor = ImageLoadActor()
        batch = pd.DataFrame([{"bytes": b"data"}])  # no 'path' column
        result = actor(batch)
        assert len(result) == 0

    def test_corrupt_row_skipped(self) -> None:
        actor = ImageLoadActor()
        good = _image_to_bytes(_make_image(5, 5))
        batch = pd.DataFrame(
            [
                {"bytes": b"corrupt", "path": "/bad.png"},
                {"bytes": good, "path": "/good.png"},
            ]
        )
        result = actor(batch)
        # Corrupt row produces an error record, good row succeeds.
        assert len(result) == 2


# -- Tests: SUPPORTED_IMAGE_EXTENSIONS ----------------------------------------


def test_supported_extensions() -> None:
    expected = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".svg"}
    assert SUPPORTED_IMAGE_EXTENSIONS == expected
