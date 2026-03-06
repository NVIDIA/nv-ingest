# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the ``nr`` CLI entrypoint, the ``nr ingest`` command group,
and the ``nr ingest inprocess`` / ``nr ingest batch`` subcommands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from nemo_retriever.adapters.cli.main import app
except ImportError as _exc:
    pytest.skip(f"CLI dependencies not available: {_exc}", allow_module_level=True)

from typer.testing import CliRunner

RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# pyproject.toml entrypoint
# ---------------------------------------------------------------------------


def test_pyproject_entrypoint_is_nr() -> None:
    """The installed script name must be ``nr``, not ``retriever``."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text()
    assert "\nnr = " in text
    assert "\nretriever = " not in text


# ---------------------------------------------------------------------------
# Top-level app
# ---------------------------------------------------------------------------


def test_top_level_help_succeeds() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_top_level_help_lists_ingest() -> None:
    result = RUNNER.invoke(app, ["--help"])
    assert "ingest" in result.output


def test_version_flag() -> None:
    result = RUNNER.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_existing_subcommands_still_registered() -> None:
    """All pre-existing subcommands must still appear in top-level help."""
    result = RUNNER.invoke(app, ["--help"])
    for name in ("audio", "pdf", "local", "harness", "recall", "benchmark"):
        assert name in result.output, f"Expected subcommand {name!r} in top-level help"


# ---------------------------------------------------------------------------
# ``nr ingest`` command group
# ---------------------------------------------------------------------------


def test_ingest_help_succeeds() -> None:
    result = RUNNER.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0


def test_ingest_help_lists_batch_and_inprocess() -> None:
    result = RUNNER.invoke(app, ["ingest", "--help"])
    assert "batch" in result.output
    assert "inprocess" in result.output


# ---------------------------------------------------------------------------
# ``nr ingest batch`` subcommand
# ---------------------------------------------------------------------------


def test_ingest_batch_help_succeeds() -> None:
    result = RUNNER.invoke(app, ["ingest", "batch", "--help"])
    assert result.exit_code == 0


def test_ingest_batch_help_shows_key_options() -> None:
    result = RUNNER.invoke(app, ["ingest", "batch", "--help"])
    for flag in ("--embed-model-name", "--lancedb-uri", "--input-type", "--debug"):
        assert flag in result.output, f"Expected {flag!r} in batch --help output"


# ---------------------------------------------------------------------------
# ``nr ingest inprocess`` subcommand – help / option presence
# ---------------------------------------------------------------------------


def test_ingest_inprocess_help_succeeds() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess", "--help"])
    assert result.exit_code == 0


def test_ingest_inprocess_help_shows_extract_options() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess", "--help"])
    for flag in (
        "--use-table-structure",
        "--table-output-format",
        "--table-structure-invoke-url",
        "--page-elements-invoke-url",
        "--ocr-invoke-url",
    ):
        assert flag in result.output, f"Expected {flag!r} in inprocess --help"


def test_ingest_inprocess_help_shows_embed_options() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess", "--help"])
    for flag in (
        "--embed-model-name",
        "--embed-invoke-url",
        "--embed-modality",
        "--embed-granularity",
        "--text-elements-modality",
        "--structured-elements-modality",
    ):
        assert flag in result.output, f"Expected {flag!r} in inprocess --help"


def test_ingest_inprocess_help_shows_execution_options() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess", "--help"])
    for flag in ("--max-workers", "--gpu-devices", "--num-gpus"):
        assert flag in result.output, f"Expected {flag!r} in inprocess --help"


def test_ingest_inprocess_help_shows_recall_options() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess", "--help"])
    for flag in ("--query-csv", "--no-recall-details"):
        assert flag in result.output, f"Expected {flag!r} in inprocess --help"


# ---------------------------------------------------------------------------
# ``nr ingest inprocess`` subcommand – argument validation
# ---------------------------------------------------------------------------


def test_ingest_inprocess_rejects_missing_path() -> None:
    result = RUNNER.invoke(app, ["ingest", "inprocess"])
    assert result.exit_code != 0
