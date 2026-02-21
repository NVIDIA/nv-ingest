"""
Txt ingestion: tokenizer-based split and chunk DataFrame builder.

Compatible with the same embed and LanceDB stages as PDF primitives.
"""

from .split import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TOKENIZER_MODEL_ID,
    split_text_by_tokens,
    txt_file_to_chunks_df,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "DEFAULT_TOKENIZER_MODEL_ID",
    "split_text_by_tokens",
    "txt_file_to_chunks_df",
]
