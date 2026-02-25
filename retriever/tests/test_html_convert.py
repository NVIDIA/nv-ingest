"""
Unit tests for retriever.html.convert: html_to_markdown, html_file_to_chunks_df, html_bytes_to_chunks_df.
"""

from pathlib import Path

import pandas as pd
import pytest

from retriever.html.convert import html_bytes_to_chunks_df, html_file_to_chunks_df, html_to_markdown


def test_html_to_markdown_str():
    pytest.importorskip("markitdown")
    html = "<html><body><p>Hello world</p></body></html>"
    md = html_to_markdown(html)
    assert isinstance(md, str)
    assert "Hello" in md or "world" in md


def test_html_to_markdown_bytes():
    pytest.importorskip("markitdown")
    html = b"<html><body><h1>Title</h1></body></html>"
    md = html_to_markdown(html)
    assert isinstance(md, str)
    assert "Title" in md


def test_html_to_markdown_path(tmp_path: Path):
    pytest.importorskip("markitdown")
    f = tmp_path / "page.html"
    f.write_text("<html><body><p>From file</p></body></html>", encoding="utf-8")
    md = html_to_markdown(str(f))
    assert isinstance(md, str)
    assert "From" in md or "file" in md


def test_html_file_to_chunks_df(tmp_path: Path):
    pytest.importorskip("markitdown")
    pytest.importorskip("transformers")
    f = tmp_path / "doc.html"
    f.write_text(
        "<html><body><h1>Heading</h1><p>First paragraph.</p><p>Second paragraph.</p></body></html>",
        encoding="utf-8",
    )
    df = html_file_to_chunks_df(
        str(f),
        max_tokens=512,
        overlap_tokens=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns and "path" in df.columns and "page_number" in df.columns and "metadata" in df.columns
    assert len(df) >= 1
    assert df["path"].iloc[0] == str(f.resolve())
    assert df["page_number"].iloc[0] >= 1
    assert "source_path" in df["metadata"].iloc[0]
    assert "chunk_index" in df["metadata"].iloc[0]
    assert df["text"].iloc[0].strip()


def test_html_file_to_chunks_df_empty_content(tmp_path: Path):
    pytest.importorskip("markitdown")
    pytest.importorskip("transformers")
    f = tmp_path / "empty.html"
    f.write_text("<html><body></body></html>", encoding="utf-8")
    df = html_file_to_chunks_df(str(f), max_tokens=512)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "path", "page_number", "metadata"]
    assert len(df) == 0


def test_html_bytes_to_chunks_df(tmp_path: Path):
    pytest.importorskip("markitdown")
    pytest.importorskip("transformers")
    html_bytes = b"<html><body><p>Chunk content from bytes.</p></body></html>"
    path = str(tmp_path / "virtual.html")
    df = html_bytes_to_chunks_df(html_bytes, path, max_tokens=512, overlap_tokens=0)
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns and "path" in df.columns and "page_number" in df.columns and "metadata" in df.columns
    assert len(df) >= 1
    assert df["path"].iloc[0] == path
    assert "source_path" in df["metadata"].iloc[0]
    assert df["text"].iloc[0].strip()
