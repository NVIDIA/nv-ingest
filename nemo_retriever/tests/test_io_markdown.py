import json
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.io import to_markdown, to_markdown_by_page


class _LazyRows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


def test_to_markdown_renders_page_dataframe() -> None:
    df = pd.DataFrame(
        [
            {
                "page_number": 1,
                "text": "Executive summary",
                "table": [{"text": "| Animal | Count |\n| --- | --- |\n| Cat | 2 |"}],
                "chart": [{"text": "Quarterly growth remained positive."}],
                "infographic": [],
            },
            {
                "page_number": 2,
                "text": "Appendix",
                "table": [],
                "chart": [],
                "infographic": [{"text": "Icon legend and callouts."}],
            },
        ]
    )

    markdown = to_markdown(df)

    assert markdown.startswith("# Extracted Content")
    assert "## Page 1" in markdown
    assert "Executive summary" in markdown
    assert "### Table 1" in markdown
    assert "### Chart 1" in markdown
    assert "## Page 2" in markdown
    assert "### Infographic 1" in markdown


def test_to_markdown_by_page_sorts_pages_and_groups_unknown() -> None:
    pages = to_markdown_by_page(
        [
            {"page_number": "2", "text": "Second page"},
            {"page_number": None, "text": "Unknown page"},
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"},
        ]
    )

    assert list(pages.keys()) == [1, 2, -1]
    assert pages[1].startswith("## Page 1")
    assert pages[2].count("Second page") == 1
    assert pages[-1].startswith("## Page Unknown")


def test_to_markdown_supports_primitive_rows_from_lazy_iterable() -> None:
    rows = _LazyRows(
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "Page text",
                    "content_metadata": {"page_number": 1},
                },
            },
            {
                "document_type": "structured",
                "metadata": {
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {"table_content": "| A |\n| --- |\n| 1 |"},
                },
            },
            {
                "document_type": "image",
                "metadata": {
                    "content_metadata": {"page_number": 2, "subtype": "page_image"},
                    "image_metadata": {"text": "OCR fallback"},
                },
            },
        ]
    )

    pages = to_markdown_by_page(rows)

    assert "Page text" in pages[1]
    assert "### Table 1" in pages[1]
    assert "### Page Image 1" in pages[2]
    assert "OCR fallback" in pages[2]


def test_to_markdown_reads_saved_records_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    payload = {
        "records": [
            {
                "page_number": 1,
                "text": "Saved result text",
                "table": [{"text": "| H |\n| --- |\n| V |"}],
                "metadata": {"source_path": "/tmp/example.pdf"},
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    markdown = to_markdown(path)

    assert "Saved result text" in markdown
    assert "### Table 1" in markdown


def test_to_markdown_empty_results_returns_none() -> None:
    assert to_markdown([]) is None


def test_to_markdown_rejects_multi_document_results() -> None:
    doc_a = pd.DataFrame([{"page_number": 1, "text": "A"}])
    doc_b = pd.DataFrame([{"page_number": 1, "text": "B"}])

    with pytest.raises(ValueError, match="single document result"):
        to_markdown([doc_a, doc_b])
