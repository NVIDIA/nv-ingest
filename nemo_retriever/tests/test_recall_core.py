import pandas as pd
import pytest

from nemo_retriever.recall.core import _normalize_query_df, is_hit_at_k


@pytest.mark.parametrize(
    "match_mode,df,expected",
    [
        ("pdf_only", pd.DataFrame({"query": ["q1"], "expected_pdf": ["Doc_A.pdf"]}), ["Doc_A"]),
        ("pdf_page", pd.DataFrame({"query": ["q1"], "pdf": ["Doc_A.pdf"], "page": [2]}), ["Doc_A_2"]),
    ],
)
def test_normalize_query_df_modes(match_mode: str, df: pd.DataFrame, expected: list[str]) -> None:
    out = _normalize_query_df(df, match_mode=match_mode)
    assert out["golden_answer"].tolist() == expected


@pytest.mark.parametrize(
    "match_mode,golden,retrieved,k,expected",
    [
        ("pdf_only", "Doc_A", ["Doc_A_7", "Doc_B_1"], 1, True),
        ("pdf_only", "Doc_B", ["Doc_A_7", "Doc_B_1"], 1, False),
        ("pdf_page", "Doc_A_2", ["Doc_A_2", "Doc_A_9"], 1, True),
        ("pdf_page", "Doc_A_2", ["Doc_A_9", "Doc_A_-1"], 1, False),
    ],
)
def test_is_hit_at_k_modes(match_mode: str, golden: str, retrieved: list[str], k: int, expected: bool) -> None:
    assert is_hit_at_k(golden, retrieved, k, match_mode=match_mode) is expected
