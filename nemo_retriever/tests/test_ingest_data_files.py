# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameterized integration tests that exercise the ingestor pipeline against
every eligible file in the ``data/`` directory, following the usage patterns
documented in the project README.

Two extraction paths are tested:
- ``.extract()`` for PDFs, DOCX, and PPTX files (README §5, §6, §"Render one document as markdown").
- ``.extract_image_files()`` for images — PNG, JPEG, BMP, TIFF, SVG (README §7).

Each test adds ``.embed()``, runs ingestion, loads the resulting records into
LanceDB via ``handle_lancedb``, and verifies the LanceDB record count matches
the number of embedded chunks produced by the ingestor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import lancedb
import pandas as pd
import pytest

from nemo_retriever import create_ingestor
from nemo_retriever.io import to_markdown
from nemo_retriever.params import ExtractParams
from nemo_retriever.vector_store.lancedb_store import (
    _build_lancedb_rows_from_df,
    handle_lancedb,
)
import tempfile


# A pytest fixture to create a temporary, isolated LanceDB instance for each test
@pytest.fixture
def db_path():
    # Use tempfile to create a temporary directory that is automatically managed
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup the directory after the test finishes
    shutil.rmtree(temp_dir)


@pytest.fixture
def db_client(db_path):
    # Connect to the temporary database path
    return lancedb.connect(db_path)


DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# ---------------------------------------------------------------------------
# File categorisation
# ---------------------------------------------------------------------------
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}
IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".svg"}
TEXT_EXTENSIONS = {".txt"}
HTML_EXTENSIONS = {".html"}

INGESTABLE_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS | HTML_EXTENSIONS


def _collect_files() -> list[Path]:
    """Return sorted list of ingestable files in the data directory."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in INGESTABLE_EXTENSIONS)


def _file_id(path: Path) -> str:
    """Readable pytest ID from a Path."""
    return path.name


ALL_FILES = _collect_files()

DOCUMENT_FILES = [p for p in ALL_FILES if p.suffix.lower() in DOCUMENT_EXTENSIONS]
IMAGE_FILES = [p for p in ALL_FILES if p.suffix.lower() in IMAGE_EXTENSIONS]
TEXT_FILES = [p for p in ALL_FILES if p.suffix.lower() in TEXT_EXTENSIONS]
HTML_FILES = [p for p in ALL_FILES if p.suffix.lower() in HTML_EXTENSIONS]

EXTRACT_PARAMS = ExtractParams(
    extract_text=True,
    extract_tables=True,
    extract_charts=True,
    extract_infographics=True,
)

LANCEDB_TABLE = "test_ingest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _results_to_rows(results: list) -> list[dict]:
    """Flatten inprocess results (list of DataFrames) into a list of row dicts."""
    all_rows: list[dict] = []
    for item in results:
        if isinstance(item, pd.DataFrame) and not item.empty:
            all_rows.extend(item.to_dict(orient="records"))
    return all_rows


def _count_embedded_rows(rows: list[dict]) -> int:
    """Count rows that have a non-None embedding in metadata (same filter as LanceDB writer)."""
    return len(_build_lancedb_rows_from_df(rows))


def _run_and_verify_lancedb(results: list, tmp_path: Path, filename: str) -> None:
    """Write results to LanceDB and verify record count matches embedded chunks."""
    assert results is not None, f"ingest() returned None for {filename}"
    assert len(results) > 0, f"ingest() returned empty results for {filename}"

    # Markdown rendering sanity check.
    md = to_markdown(results[0])
    assert isinstance(md, str), f"to_markdown did not return a string for {filename}"

    # Flatten results into rows and count those that carry embeddings.
    rows = _results_to_rows(results)
    expected_count = _count_embedded_rows(rows)
    assert expected_count > 0, f"No embedded rows produced for {filename}"

    # Write to LanceDB via handle_lancedb (same path as batch_pipeline).
    lancedb_uri = str(tmp_path / "lancedb")
    handle_lancedb(rows, uri=lancedb_uri, table_name=LANCEDB_TABLE)

    # Open the table and verify the record count.
    db = lancedb.connect(uri=lancedb_uri)
    table = db.open_table(LANCEDB_TABLE)
    actual_count = table.count_rows()

    assert actual_count == expected_count, (
        f"LanceDB record count mismatch for {filename}: " f"expected {expected_count}, got {actual_count}"
    )


# ---------------------------------------------------------------------------
# Document extraction tests (PDF, DOCX, PPTX) — mirrors README §5 & markdown
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filepath", DOCUMENT_FILES, ids=_file_id)
def test_extract_document(filepath: Path, tmp_path: Path) -> None:
    """Ingest a document via ``.extract().embed()`` and verify LanceDB record count."""
    ingestor = create_ingestor(run_mode="inprocess").files(str(filepath)).extract(EXTRACT_PARAMS).embed()
    results = ingestor.ingest()
    _run_and_verify_lancedb(results, tmp_path, filepath.name)


# ---------------------------------------------------------------------------
# Image extraction tests (PNG, JPEG, BMP, TIFF, SVG) — mirrors README §7
# ---------------------------------------------------------------------------
_has_cairosvg = True
try:
    import cairosvg  # noqa: F401
except ImportError:
    _has_cairosvg = False


@pytest.mark.parametrize("filepath", IMAGE_FILES, ids=_file_id)
def test_extract_image(filepath: Path, tmp_path: Path) -> None:
    """Ingest an image via ``.extract_image_files().embed()`` and verify LanceDB record count."""
    if filepath.suffix.lower() == ".svg" and not _has_cairosvg:
        pytest.skip("cairosvg is required for SVG support")

    ingestor = create_ingestor(run_mode="inprocess").files(str(filepath)).extract_image_files(EXTRACT_PARAMS).embed()
    results = ingestor.ingest()
    _run_and_verify_lancedb(results, tmp_path, filepath.name)


# ---------------------------------------------------------------------------
# Text file extraction — mirrors README §6 (input-type txt)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filepath", TEXT_FILES, ids=_file_id)
def test_extract_text(filepath: Path, tmp_path: Path) -> None:
    """Ingest a text file via ``.extract().embed()`` and verify LanceDB record count."""
    ingestor = create_ingestor(run_mode="inprocess").files(str(filepath)).extract(EXTRACT_PARAMS).embed()
    results = ingestor.ingest()
    _run_and_verify_lancedb(results, tmp_path, filepath.name)


# ---------------------------------------------------------------------------
# HTML file extraction — mirrors README §6 (input-type html)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filepath", HTML_FILES, ids=_file_id)
def test_extract_html(filepath: Path, tmp_path: Path) -> None:
    """Ingest an HTML file via ``.extract().embed()`` and verify LanceDB record count."""
    ingestor = create_ingestor(run_mode="inprocess").files(str(filepath)).extract(EXTRACT_PARAMS).embed()
    results = ingestor.ingest()
    _run_and_verify_lancedb(results, tmp_path, filepath.name)
