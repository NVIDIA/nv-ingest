import pandas as pd

from nemo_retriever.recall.core import _is_hit, _normalize_query_df


def test_normalize_query_df_pdf_only_uses_expected_pdf() -> None:
    df = pd.DataFrame({"query": ["q1"], "expected_pdf": ["Doc_A.pdf"]})
    out = _normalize_query_df(df, match_mode="pdf_only")
    assert out["golden_answer"].tolist() == ["Doc_A"]


def test_normalize_query_df_pdf_page_uses_pdf_and_page() -> None:
    df = pd.DataFrame({"query": ["q1"], "pdf": ["Doc_A.pdf"], "page": [2]})
    out = _normalize_query_df(df, match_mode="pdf_page")
    assert out["golden_answer"].tolist() == ["Doc_A_2"]


def test_is_hit_pdf_only_matches_any_page_of_document() -> None:
    retrieved = ["Doc_A_7", "Doc_B_1"]
    assert _is_hit("Doc_A", retrieved, 1, match_mode="pdf_only") is True
    assert _is_hit("Doc_B", retrieved, 1, match_mode="pdf_only") is False
