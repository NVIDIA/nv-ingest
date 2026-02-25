"""
Unit tests for retriever.txt.split: split_text_by_tokens and txt_file_to_chunks_df.
"""

import tempfile  # noqa: F401
from pathlib import Path

import pandas as pd
import pytest

from retriever.txt.split import split_text_by_tokens, txt_file_to_chunks_df


class _MockTokenizer:
    """Minimal tokenizer: encode = split on spaces, decode = join."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, (list, range)):
            return " ".join(str(i) for i in ids)
        return str(ids)


def test_split_text_by_tokens_empty():
    tokenizer = _MockTokenizer()
    assert split_text_by_tokens("", tokenizer=tokenizer, max_tokens=10) == []
    assert split_text_by_tokens("   \n  ", tokenizer=tokenizer, max_tokens=10) == []


def test_split_text_by_tokens_no_overlap():
    tokenizer = _MockTokenizer()
    # "a b c d e f g h i j" -> 10 tokens, max_tokens=3 -> 4 chunks
    text = "a b c d e f g h i j"
    chunks = split_text_by_tokens(text, tokenizer=tokenizer, max_tokens=3, overlap_tokens=0)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "a" in joined and "j" in joined


def test_split_text_by_tokens_single_chunk():
    tokenizer = _MockTokenizer()
    text = "one two three"
    chunks = split_text_by_tokens(text, tokenizer=tokenizer, max_tokens=10, overlap_tokens=0)
    assert len(chunks) == 1
    assert chunks[0] == "one two three"


def test_split_text_by_tokens_max_tokens_positive():
    tokenizer = _MockTokenizer()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        split_text_by_tokens("hello", tokenizer=tokenizer, max_tokens=0)


def test_txt_file_to_chunks_df(tmp_path: Path):
    pytest.importorskip("transformers")
    f = tmp_path / "doc.txt"
    f.write_text("First paragraph here. Second paragraph there.", encoding="utf-8")
    df = txt_file_to_chunks_df(
        str(f),
        max_tokens=512,
        overlap_tokens=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "path", "page_number", "metadata"]
    assert len(df) >= 1
    assert df["path"].iloc[0] == str(f.resolve())
    assert df["page_number"].iloc[0] >= 1
    assert "source_path" in df["metadata"].iloc[0]
    assert "chunk_index" in df["metadata"].iloc[0]


def test_txt_file_to_chunks_df_empty_file(tmp_path: Path):
    pytest.importorskip("transformers")
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    df = txt_file_to_chunks_df(str(f), max_tokens=512)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "path", "page_number", "metadata"]
    assert len(df) == 0
