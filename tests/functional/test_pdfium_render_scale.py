# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Edge case tests for _compute_render_scale_to_fit.

These tests guard against degenerate inputs that could cause division by zero
or other failures. They do NOT validate correctness of the scaling math -
that requires integration tests comparing actual rendered output.
"""

from unittest.mock import MagicMock

from nv_ingest_api.util.pdf.pdfium import _compute_render_scale_to_fit


def _make_mock_page(width: float, height: float) -> MagicMock:
    """Create a mock PdfPage with given dimensions."""
    page = MagicMock()
    page.get_width.return_value = width
    page.get_height.return_value = height
    return page


class TestComputeRenderScaleToFitEdgeCases:
    """Edge case guards for _compute_render_scale_to_fit."""

    def test_zero_target_width_returns_default(self):
        """Zero target width should return 1.0 to avoid division by zero."""
        page = _make_mock_page(100, 100)
        scale = _compute_render_scale_to_fit(page, (0, 100))
        assert scale == 1.0

    def test_zero_target_height_returns_default(self):
        """Zero target height should return 1.0 to avoid division by zero."""
        page = _make_mock_page(100, 100)
        scale = _compute_render_scale_to_fit(page, (100, 0))
        assert scale == 1.0

    def test_zero_page_width_returns_default(self):
        """Zero page width should return 1.0 to avoid division by zero."""
        page = _make_mock_page(0, 100)
        scale = _compute_render_scale_to_fit(page, (100, 100))
        assert scale == 1.0

    def test_zero_page_height_returns_default(self):
        """Zero page height should return 1.0 to avoid division by zero."""
        page = _make_mock_page(100, 0)
        scale = _compute_render_scale_to_fit(page, (100, 100))
        assert scale == 1.0

    def test_negative_target_returns_default(self):
        """Negative target dimensions should return 1.0."""
        page = _make_mock_page(100, 100)
        scale = _compute_render_scale_to_fit(page, (-100, 100))
        assert scale == 1.0

    def test_minimum_scale_clamp(self):
        """Scale should be clamped to minimum of 1e-3 to prevent degenerate renders."""
        page = _make_mock_page(1000000, 1000000)
        scale = _compute_render_scale_to_fit(page, (1, 1))
        assert scale >= 1e-3
