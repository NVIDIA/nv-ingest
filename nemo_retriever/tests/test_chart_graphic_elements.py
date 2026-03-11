# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the combined graphic-elements + OCR chart stage."""

from __future__ import annotations

import base64
import importlib
import io
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nemo_retriever.utils.table_and_chart import join_graphic_elements_and_ocr_output


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
# join_graphic_elements_and_ocr_output tests
# ---------------------------------------------------------------------------


class TestJoinGraphicElementsAndOCR:
    """Test the core joining function with synthetic data."""

    def test_empty_ocr_returns_empty(self) -> None:
        ge_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5], "label_name": "chart_title"},
        ]
        result = join_graphic_elements_and_ocr_output(ge_dets, [], (100, 200))
        assert result == ""

    def test_no_ge_dets_returns_empty(self) -> None:
        ocr_preds = [
            {"left": 0.0, "right": 0.5, "upper": 0.0, "lower": 0.5, "text": "hello"},
        ]
        result = join_graphic_elements_and_ocr_output([], ocr_preds, (100, 200))
        assert result == ""

    def test_matching_ge_and_ocr(self) -> None:
        ge_dets = [
            {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 0.3], "label_name": "chart_title"},
            {"bbox_xyxy_norm": [0.0, 0.7, 1.0, 1.0], "label_name": "xlabel"},
        ]
        ocr_preds = [
            {"left": 0.1, "right": 0.9, "upper": 0.05, "lower": 0.25, "text": "Sales Chart"},
            {"left": 0.1, "right": 0.9, "upper": 0.75, "lower": 0.95, "text": "Quarter"},
        ]
        result = join_graphic_elements_and_ocr_output(ge_dets, ocr_preds, (100, 200))
        assert result  # Non-empty
        assert "Sales Chart" in result or "Quarter" in result


# ---------------------------------------------------------------------------
# graphic_elements_ocr_page_elements tests
# ---------------------------------------------------------------------------


def _make_chart_page_df(
    width: int = 200,
    height: int = 100,
    has_chart: bool = True,
) -> pd.DataFrame:
    """Build a single-row page DataFrame with a chart detection."""
    image_b64 = _make_b64_png(width, height)
    detections = []
    if has_chart:
        detections.append(
            {
                "label_name": "chart",
                "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                "score": 0.95,
            }
        )
    return pd.DataFrame(
        [
            {
                "page_image": {"image_b64": image_b64},
                "page_elements_v3": {"detections": detections},
                "page_elements_v3_counts_by_label": {"chart": len(detections)},
            }
        ]
    )


@_needs_pil
@_needs_requests
@_needs_torch
class TestGraphicElementsOCRPageElements:
    """Test the full graphic_elements_ocr_page_elements function with mocked models."""

    def test_no_charts_produces_empty_chart_column(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = _make_chart_page_df(has_chart=False)
        mock_ge_model = MagicMock()
        mock_ocr_model = MagicMock()

        result = graphic_elements_ocr_page_elements(
            df,
            graphic_elements_model=mock_ge_model,
            ocr_model=mock_ocr_model,
        )
        assert "chart" in result.columns
        assert "graphic_elements_ocr_v1" in result.columns
        assert result.iloc[0]["chart"] == []
        mock_ge_model.invoke.assert_not_called()
        mock_ocr_model.invoke.assert_not_called()

    def test_no_page_image_produces_empty_chart_column(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = pd.DataFrame(
            [
                {
                    "page_image": {},
                    "page_elements_v3": {
                        "detections": [
                            {"label_name": "chart", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
                        ]
                    },
                }
            ]
        )
        mock_ge_model = MagicMock()
        mock_ocr_model = MagicMock()

        result = graphic_elements_ocr_page_elements(
            df,
            graphic_elements_model=mock_ge_model,
            ocr_model=mock_ocr_model,
        )
        assert result.iloc[0]["chart"] == []

    def test_with_mocked_models_produces_text(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        # Mock GE model: returns chart_title detection
        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title", "xlabel", "ylabel"]

        mock_pred = {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 0.3]]),
            "labels": torch.tensor([0]),  # chart_title
            "scores": torch.tensor([0.9]),
        }
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.return_value = mock_pred

        # Mock OCR model
        mock_ocr_model = MagicMock()
        mock_ocr_model.invoke.return_value = [
            {"left": 0.1, "right": 0.9, "upper": 0.05, "lower": 0.25, "text": "ChartTitle"},
        ]

        result = graphic_elements_ocr_page_elements(
            df,
            graphic_elements_model=mock_ge_model,
            ocr_model=mock_ocr_model,
        )

        assert "chart" in result.columns
        chart_entries = result.iloc[0]["chart"]
        assert len(chart_entries) == 1
        assert "ChartTitle" in chart_entries[0]["text"]
        assert chart_entries[0]["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]

    def test_fallback_when_no_ge_detections(self) -> None:
        """When GE model returns no detections, should fall back to OCR-only text."""
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        # Mock GE model: returns no detections
        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title"]

        mock_pred = {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.return_value = mock_pred

        # Mock OCR model
        mock_ocr_model = MagicMock()
        mock_ocr_model.invoke.return_value = [
            {"left": 0.1, "right": 0.9, "upper": 0.1, "lower": 0.4, "text": "FallbackText"},
        ]

        result = graphic_elements_ocr_page_elements(
            df,
            graphic_elements_model=mock_ge_model,
            ocr_model=mock_ocr_model,
        )

        chart_entries = result.iloc[0]["chart"]
        assert len(chart_entries) == 1
        assert "FallbackText" in chart_entries[0]["text"]

    def test_model_error_recorded_in_metadata(self) -> None:
        """When model raises an exception, it should be recorded in metadata, not crash."""
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        import torch

        df = _make_chart_page_df(width=200, height=100)

        mock_ge_model = MagicMock()
        mock_ge_model._model = MagicMock()
        mock_ge_model._model.labels = ["chart_title"]
        mock_ge_model.preprocess.return_value = torch.zeros(1, 3, 100, 200)
        mock_ge_model.invoke.side_effect = RuntimeError("model exploded")

        mock_ocr_model = MagicMock()

        result = graphic_elements_ocr_page_elements(
            df,
            graphic_elements_model=mock_ge_model,
            ocr_model=mock_ocr_model,
        )

        meta = result.iloc[0]["graphic_elements_ocr_v1"]
        assert meta["error"] is not None
        assert meta["error"]["type"] == "RuntimeError"
        assert "model exploded" in meta["error"]["message"]

    def test_requires_model_when_no_url(self) -> None:
        from nemo_retriever.chart.chart_detection import graphic_elements_ocr_page_elements

        df = _make_chart_page_df()
        with pytest.raises(ValueError, match="graphic_elements_model"):
            graphic_elements_ocr_page_elements(df, ocr_model=MagicMock())

        with pytest.raises(ValueError, match="ocr_model"):
            graphic_elements_ocr_page_elements(df, graphic_elements_model=MagicMock())


# ---------------------------------------------------------------------------
# GraphicElementsActor tests
# ---------------------------------------------------------------------------


@_needs_pil
class TestGraphicElementsActor:
    """Test the Ray actor wrapper."""

    def test_actor_error_returns_dataframe_with_error(self) -> None:
        """Actor should never raise; errors go into metadata columns."""
        from nemo_retriever.chart.chart_detection import GraphicElementsActor

        actor = GraphicElementsActor.__new__(GraphicElementsActor)
        actor._graphic_elements_model = None
        actor._ocr_model = None
        actor._graphic_elements_invoke_url = ""
        actor._ocr_invoke_url = ""
        actor._api_key = None
        actor._request_timeout_s = 120.0
        actor._remote_retry = None

        df = _make_chart_page_df()
        # This will fail because both models are None and no URLs set.
        result = actor(df)
        assert "chart" in result.columns
        assert "graphic_elements_ocr_v1" in result.columns
        meta = result.iloc[0]["graphic_elements_ocr_v1"]
        assert meta["error"] is not None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


@_needs_cv2
class TestGraphicElementsOCRConfig:
    def test_load_config_defaults(self) -> None:
        from nemo_retriever.chart.config import load_graphic_elements_ocr_config_from_dict

        cfg = load_graphic_elements_ocr_config_from_dict({})
        assert cfg.graphic_elements_invoke_url == ""
        assert cfg.ocr_invoke_url == ""
        assert cfg.api_key == ""
        assert cfg.request_timeout_s == 120.0

    def test_load_config_with_values(self) -> None:
        from nemo_retriever.chart.config import load_graphic_elements_ocr_config_from_dict

        cfg = load_graphic_elements_ocr_config_from_dict(
            {
                "graphic_elements_invoke_url": "http://ge:8000",
                "ocr_invoke_url": "http://ocr:8000",
                "api_key": "secret",
                "request_timeout_s": 60.0,
            }
        )
        assert cfg.graphic_elements_invoke_url == "http://ge:8000"
        assert cfg.ocr_invoke_url == "http://ocr:8000"
        assert cfg.api_key == "secret"
        assert cfg.request_timeout_s == 60.0


# ---------------------------------------------------------------------------
# build_plan tests
# ---------------------------------------------------------------------------


@_needs_cv2
class TestBuildPlanChartStructure:
    def test_use_graphic_elements_selects_chart_structure_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(
                extract_charts=True,
                use_graphic_elements=True,
            )
        )
        assert "enrich_graphic_elements" in names
        assert "enrich_chart" not in names

    def test_no_graphic_elements_selects_default_chart_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_charts=True))
        assert "enrich_chart" in names
        assert "enrich_graphic_elements" not in names

    def test_no_extract_charts_yields_no_chart_stage(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(stage_names_from_flags(extract_charts=False, use_graphic_elements=True))
        assert "enrich_graphic_elements" not in names
        assert "enrich_chart" not in names

    def test_graphic_elements_flag_does_not_affect_table_stages(self) -> None:
        from nemo_retriever.application.pipeline.build_plan import stage_names_from_flags

        names = list(
            stage_names_from_flags(
                extract_tables=True,
                extract_charts=True,
                use_graphic_elements=True,
                use_table_structure=True,
                table_output_format="markdown",
            )
        )
        assert "enrich_table_structure" in names
        assert "enrich_graphic_elements" in names


# ---------------------------------------------------------------------------
# _prediction_to_detections string labels test
# ---------------------------------------------------------------------------


@_needs_torch
class TestPredictionToDetectionsStringLabels:
    def test_string_labels_handled(self) -> None:
        import torch
        from nemo_retriever.chart.chart_detection import _prediction_to_detections

        pred = {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "labels": ["chart_title", "xlabel"],
            "scores": torch.tensor([0.9, 0.8]),
        }
        dets = _prediction_to_detections(pred, label_names=[])
        assert len(dets) == 2
        assert dets[0]["label_name"] == "chart_title"
        assert dets[1]["label_name"] == "xlabel"

    def test_integer_labels_still_work(self) -> None:
        import torch
        from nemo_retriever.chart.chart_detection import _prediction_to_detections

        pred = {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        }
        dets = _prediction_to_detections(pred, label_names=["chart_title", "xlabel"])
        assert len(dets) == 1
        assert dets[0]["label_name"] == "xlabel"
        assert dets[0]["label"] == 1
