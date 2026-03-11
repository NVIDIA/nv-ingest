# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the combined table-structure + OCR stage."""

from __future__ import annotations

import base64
import importlib
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_retriever.utils.table_and_chart import join_table_structure_and_ocr_output


def _can_import(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


_needs_pil = pytest.mark.skipif(not _can_import("PIL"), reason="PIL (Pillow) not installed")
_needs_requests = pytest.mark.skipif(not _can_import("requests"), reason="requests not installed")
_needs_torch = pytest.mark.skipif(not _can_import("torch"), reason="torch not installed")
_needs_cv2 = pytest.mark.skipif(not _can_import("cv2"), reason="cv2 (opencv) not installed")


def _make_b64_png(width: int = 200, height: int = 100) -> str:
    """Create a small synthetic PNG image encoded as base64."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# join_table_structure_and_ocr_output tests
# ---------------------------------------------------------------------------


class TestJoinTableStructureAndOCR:
    """Test the core joining function with synthetic data."""

    def test_empty_ocr_returns_empty(self) -> None:
        structure_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5], "label_name": "cell"},
        ]
        result = join_table_structure_and_ocr_output(structure_dets, [], (100, 200))
        assert result == ""

    def test_no_cells_returns_empty(self) -> None:
        # Only rows/columns, no cells — should return empty.
        structure_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.5], "label_name": "row"},
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 1.0], "label_name": "column"},
        ]
        ocr_preds = [
            {"left": 0.0, "right": 0.5, "upper": 0.0, "lower": 0.5, "text": "hello"},
        ]
        result = join_table_structure_and_ocr_output(structure_dets, ocr_preds, (100, 200))
        assert result == ""

    def test_simple_2x2_table(self) -> None:
        """A 2x2 table with 4 cells, 2 rows, 2 columns."""
        # Two rows, two columns
        structure_dets = [
            # Row 0
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.5], "label_name": "row"},
            # Row 1
            {"bbox_xyxy_norm": [0.0, 0.5, 1.0, 1.0], "label_name": "row"},
            # Column 0
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 1.0], "label_name": "column"},
            # Column 1
            {"bbox_xyxy_norm": [0.5, 0.0, 1.0, 1.0], "label_name": "column"},
            # Cell (0,0)
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5], "label_name": "cell"},
            # Cell (0,1)
            {"bbox_xyxy_norm": [0.5, 0.0, 1.0, 0.5], "label_name": "cell"},
            # Cell (1,0)
            {"bbox_xyxy_norm": [0.0, 0.5, 0.5, 1.0], "label_name": "cell"},
            # Cell (1,1)
            {"bbox_xyxy_norm": [0.5, 0.5, 1.0, 1.0], "label_name": "cell"},
        ]
        # OCR output: one word per cell, normalized coords
        ocr_preds = [
            {"left": 0.05, "right": 0.45, "upper": 0.05, "lower": 0.45, "text": "A"},
            {"left": 0.55, "right": 0.95, "upper": 0.05, "lower": 0.45, "text": "B"},
            {"left": 0.05, "right": 0.45, "upper": 0.55, "lower": 0.95, "text": "C"},
            {"left": 0.55, "right": 0.95, "upper": 0.55, "lower": 0.95, "text": "D"},
        ]
        crop_hw = (100, 200)
        result = join_table_structure_and_ocr_output(structure_dets, ocr_preds, crop_hw)
        assert result  # Should produce non-empty markdown
        # The markdown table should contain all four cell values.
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" in result
        # Should have pipe separators (markdown table format).
        assert "|" in result

    def test_ocr_dict_format(self) -> None:
        """Test with OCR output in dict format (boxes/texts keys)."""
        structure_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.5], "label_name": "row"},
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0], "label_name": "column"},
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.5], "label_name": "cell"},
        ]
        ocr_preds = {
            "boxes": [[[10, 5], [90, 5], [90, 45], [10, 45]]],
            "texts": ["Hello"],
        }
        crop_hw = (100, 200)
        result = join_table_structure_and_ocr_output(structure_dets, ocr_preds, crop_hw)
        assert "Hello" in result


# ---------------------------------------------------------------------------
# table_structure_ocr_page_elements tests
# ---------------------------------------------------------------------------


def _make_page_df(
    width: int = 200,
    height: int = 100,
    has_table: bool = True,
) -> pd.DataFrame:
    """Build a single-row page DataFrame with page_elements_v3 and page_image."""
    image_b64 = _make_b64_png(width, height)
    detections = []
    if has_table:
        detections.append(
            {
                "label_name": "table",
                "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                "score": 0.95,
            }
        )
    return pd.DataFrame(
        [
            {
                "page_image": {"image_b64": image_b64},
                "page_elements_v3": {"detections": detections},
                "page_elements_v3_counts_by_label": {"table": len(detections)},
            }
        ]
    )


@_needs_pil
@_needs_requests
@_needs_torch
class TestTableStructureOCRPageElements:
    """Test the full table_structure_ocr_page_elements function with mocked models."""

    def test_no_tables_produces_empty_table_column(self) -> None:
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df(has_table=False)
        mock_ts_model = MagicMock()
        mock_ocr_model = MagicMock()

        result = table_structure_ocr_page_elements(
            df,
            table_structure_model=mock_ts_model,
            ocr_model=mock_ocr_model,
        )
        assert "table" in result.columns
        assert "table_structure_ocr_v1" in result.columns
        assert result.iloc[0]["table"] == []
        # No model calls should have been made.
        mock_ts_model.invoke.assert_not_called()
        mock_ocr_model.invoke.assert_not_called()

    def test_no_page_image_produces_empty_table_column(self) -> None:
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        df = pd.DataFrame(
            [
                {
                    "page_image": {},
                    "page_elements_v3": {
                        "detections": [
                            {"label_name": "table", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
                        ]
                    },
                }
            ]
        )
        mock_ts_model = MagicMock()
        mock_ocr_model = MagicMock()

        result = table_structure_ocr_page_elements(
            df,
            table_structure_model=mock_ts_model,
            ocr_model=mock_ocr_model,
        )
        assert result.iloc[0]["table"] == []

    def test_with_mocked_models_produces_markdown(self) -> None:
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        import torch

        df = _make_page_df(width=200, height=100)

        # Mock table-structure model
        mock_ts_model = MagicMock()
        mock_ts_model._model = MagicMock()
        mock_ts_model._model.labels = ["cell", "row", "column"]

        # Table structure returns: 1 cell, 1 row, 1 column (full image)
        mock_pred = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([0, 1, 2]),  # cell=0, row=1, column=2
            "scores": torch.tensor([0.9, 0.9, 0.9]),
        }
        mock_ts_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ts_model.invoke.return_value = mock_pred

        # Mock OCR model
        mock_ocr_model = MagicMock()
        mock_ocr_model.invoke.return_value = [
            {"left": 0.1, "right": 0.9, "upper": 0.1, "lower": 0.9, "text": "TestValue"},
        ]

        result = table_structure_ocr_page_elements(
            df,
            table_structure_model=mock_ts_model,
            ocr_model=mock_ocr_model,
        )

        assert "table" in result.columns
        table_entries = result.iloc[0]["table"]
        assert len(table_entries) == 1
        assert "TestValue" in table_entries[0]["text"]
        assert table_entries[0]["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]

    def test_fallback_to_pseudo_markdown_when_no_cells(self) -> None:
        """When table-structure returns no cells, should fall back to pseudo-markdown."""
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        import torch

        df = _make_page_df(width=200, height=100)

        # Mock table-structure model: returns only rows and columns, no cells.
        mock_ts_model = MagicMock()
        mock_ts_model._model = MagicMock()
        mock_ts_model._model.labels = ["cell", "row", "column"]

        mock_pred = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.5, 1.0]]),
            "labels": torch.tensor([1, 2]),  # row=1, column=2 — no cells!
            "scores": torch.tensor([0.9, 0.9]),
        }
        mock_ts_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ts_model.invoke.return_value = mock_pred

        # Mock OCR model
        mock_ocr_model = MagicMock()
        mock_ocr_model.invoke.return_value = [
            {"left": 0.1, "right": 0.9, "upper": 0.1, "lower": 0.4, "text": "Row1"},
            {"left": 0.1, "right": 0.9, "upper": 0.6, "lower": 0.9, "text": "Row2"},
        ]

        result = table_structure_ocr_page_elements(
            df,
            table_structure_model=mock_ts_model,
            ocr_model=mock_ocr_model,
        )

        table_entries = result.iloc[0]["table"]
        assert len(table_entries) == 1
        text = table_entries[0]["text"]
        # Fallback pseudo-markdown should contain the OCR text.
        assert "Row1" in text
        assert "Row2" in text

    def test_model_error_recorded_in_metadata(self) -> None:
        """When model raises an exception, it should be recorded in metadata, not crash."""
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df(width=200, height=100)

        mock_ts_model = MagicMock()
        mock_ts_model._model = MagicMock()
        mock_ts_model._model.labels = ["cell", "row", "column"]
        mock_ts_model.preprocess.side_effect = RuntimeError("model exploded")

        mock_ocr_model = MagicMock()

        result = table_structure_ocr_page_elements(
            df,
            table_structure_model=mock_ts_model,
            ocr_model=mock_ocr_model,
        )

        meta = result.iloc[0]["table_structure_ocr_v1"]
        assert meta["error"] is not None
        assert meta["error"]["type"] == "RuntimeError"
        assert "model exploded" in meta["error"]["message"]

    def test_requires_model_when_no_url(self) -> None:
        from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

        df = _make_page_df()
        with pytest.raises(ValueError, match="table_structure_model"):
            table_structure_ocr_page_elements(df, ocr_model=MagicMock())

        with pytest.raises(ValueError, match="ocr_model"):
            table_structure_ocr_page_elements(df, table_structure_model=MagicMock())


# ---------------------------------------------------------------------------
# TableStructureActor tests
# ---------------------------------------------------------------------------


@_needs_pil
class TestTableStructureActor:
    """Test the Ray actor wrapper."""

    def test_actor_error_returns_dataframe_with_error(self) -> None:
        """Actor should never raise; errors go into metadata columns."""
        from nemo_retriever.table.table_detection import TableStructureActor

        # Patch model constructors to avoid loading real models.
        with (patch("nemo_retriever.table.table_detection.TableStructureActor.__init__", return_value=None),):
            actor = TableStructureActor.__new__(TableStructureActor)
            actor._table_structure_model = None
            actor._ocr_model = None
            actor._table_structure_invoke_url = ""
            actor._ocr_invoke_url = ""
            actor._api_key = None
            actor._request_timeout_s = 120.0
            actor._remote_retry = None

            df = _make_page_df()
            # This will fail because both models are None and no URLs set.
            result = actor(df)
            assert "table" in result.columns
            assert "table_structure_ocr_v1" in result.columns
            meta = result.iloc[0]["table_structure_ocr_v1"]
            assert meta["error"] is not None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestTableStructureOCRConfig:
    def test_load_config_defaults(self) -> None:
        from nemo_retriever.table.config import load_table_structure_ocr_config_from_dict

        cfg = load_table_structure_ocr_config_from_dict({})
        assert cfg.table_structure_invoke_url == ""
        assert cfg.ocr_invoke_url == ""
        assert cfg.api_key == ""
        assert cfg.request_timeout_s == 120.0

    def test_load_config_with_values(self) -> None:
        from nemo_retriever.table.config import load_table_structure_ocr_config_from_dict

        cfg = load_table_structure_ocr_config_from_dict(
            {
                "table_structure_invoke_url": "http://ts:8000",
                "ocr_invoke_url": "http://ocr:8000",
                "api_key": "secret",
                "request_timeout_s": 60.0,
            }
        )
        assert cfg.table_structure_invoke_url == "http://ts:8000"
        assert cfg.ocr_invoke_url == "http://ocr:8000"
        assert cfg.api_key == "secret"
        assert cfg.request_timeout_s == 60.0


# ---------------------------------------------------------------------------
# build_plan tests
# ---------------------------------------------------------------------------


@_needs_cv2
class TestBuildPlan:
    def test_use_table_structure_selects_structure_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(
                extract_tables=True,
                use_table_structure=True,
                table_output_format="markdown",
            )
        )
        assert "enrich_table_structure" in names
        assert "enrich_table" not in names

    def test_pseudo_markdown_selects_ocr_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_tables=True, table_output_format="pseudo_markdown"))
        assert "enrich_table" in names
        assert "enrich_table_structure" not in names

    def test_default_format_selects_ocr_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_tables=True))
        assert "enrich_table" in names
        assert "enrich_table_structure" not in names

    def test_no_extract_tables_yields_nothing(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(extract_tables=False, use_table_structure=True, table_output_format="markdown")
        )
        assert "enrich_table_structure" not in names
        assert "enrich_table" not in names

    def test_markdown_without_use_table_structure_raises(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        with pytest.raises(ValueError, match="use_table_structure=True"):
            list(
                stage_names_from_flags(
                    extract_tables=True,
                    use_table_structure=False,
                    table_output_format="markdown",
                )
            )

    def test_use_table_structure_with_pseudo_markdown_warns(self) -> None:
        import warnings
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            names = list(
                stage_names_from_flags(
                    extract_tables=True,
                    use_table_structure=True,
                    table_output_format="pseudo_markdown",
                )
            )
            assert len(w) == 1
            assert "consider using table_output_format='markdown'" in str(w[0].message)
        assert "enrich_table_structure" in names


# ---------------------------------------------------------------------------
# ExtractParams tests
# ---------------------------------------------------------------------------


class TestExtractParams:
    def test_new_fields_exist(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams(
            use_table_structure=True,
            table_output_format="markdown",
            table_structure_invoke_url="http://ts:8000",
        )
        assert params.use_table_structure is True
        assert params.table_output_format == "markdown"
        assert params.table_structure_invoke_url == "http://ts:8000"

    def test_defaults(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams()
        assert params.use_table_structure is False
        assert params.table_output_format == "pseudo_markdown"
        assert params.table_structure_invoke_url is None

    def test_auto_enable_when_invoke_url_provided(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams(table_structure_invoke_url="http://ts:8000")
        assert params.use_table_structure is True
        assert params.table_output_format == "markdown"

    def test_no_auto_enable_when_invoke_url_absent(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams(table_structure_invoke_url=None)
        assert params.use_table_structure is False

    def test_auto_markdown_when_use_table_structure(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams(use_table_structure=True)
        assert params.table_output_format == "markdown"

    def test_explicit_pseudo_markdown_preserved(self) -> None:
        from nemo_retriever.params.models import ExtractParams

        params = ExtractParams(use_table_structure=True, table_output_format="pseudo_markdown")
        assert params.table_output_format == "pseudo_markdown"
