import json
from pathlib import Path

import pandas as pd

from nemo_retriever.io import to_markdown, to_markdown_by_page


class _LazyRows:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class _DatasetLike:
    def __init__(self, rows):
        self._rows = rows

    def take_all(self):
        return list(self._rows)


class _BatchResults:
    def __init__(self, rows):
        self._rd_dataset = _DatasetLike(rows)


def test_to_markdown_groups_page_dataframe_by_filename() -> None:
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/alpha.pdf",
                "page_number": 1,
                "text": "Executive summary",
                "table": [{"text": "| Animal | Count |\n| --- | --- |\n| Cat | 2 |"}],
                "chart": [{"text": "Quarterly growth remained positive."}],
                "infographic": [],
            },
            {
                "path": "/tmp/alpha.pdf",
                "page_number": 2,
                "text": "Appendix",
                "table": [],
                "chart": [],
                "infographic": [],
            },
            {
                "path": "/tmp/beta.pdf",
                "page_number": 1,
                "text": "Appendix",
                "table": [],
                "chart": [],
                "infographic": [{"text": "Icon legend and callouts."}],
            },
        ]
    )

    markdown = to_markdown(df)

    assert list(markdown) == ["alpha.pdf", "beta.pdf"]
    assert markdown["alpha.pdf"].startswith("Executive summary")
    assert "Executive summary" in markdown["alpha.pdf"]
    assert "### Table 1" in markdown["alpha.pdf"]
    assert "### Chart 1" in markdown["alpha.pdf"]
    assert "Appendix" in markdown["alpha.pdf"]
    assert "## Page 1" not in markdown["alpha.pdf"]
    assert "## Page 2" not in markdown["alpha.pdf"]
    assert "### Infographic 1" in markdown["beta.pdf"]


def test_to_markdown_by_page_sorts_pages_and_groups_unknown_per_document() -> None:
    pages = to_markdown_by_page(
        [
            {"source_path": "/tmp/alpha.pdf", "page_number": "2", "text": "Second page"},
            {"source_path": "/tmp/alpha.pdf", "page_number": None, "text": "Unknown page"},
            {"source_path": "/tmp/alpha.pdf", "page_number": 1, "text": "First page"},
            {"source_path": "/tmp/alpha.pdf", "page_number": 2, "text": "Second page"},
            {"source_path": "/tmp/beta.pdf", "page_number": 1, "text": "Only page"},
        ]
    )

    assert list(pages["alpha.pdf"].keys()) == [1, 2, -1]
    assert pages["alpha.pdf"][1] == "First page"
    assert pages["alpha.pdf"][2].count("Second page") == 1
    assert pages["alpha.pdf"][-1] == "Unknown page"
    assert pages["beta.pdf"][1] == "Only page"


def test_to_markdown_supports_primitive_rows_from_lazy_iterable() -> None:
    rows = _LazyRows(
        [
            {
                "document_type": "text",
                "metadata": {
                    "source_path": "/tmp/alpha.pdf",
                    "content": "Page text",
                    "content_metadata": {"page_number": 1},
                },
            },
            {
                "document_type": "structured",
                "metadata": {
                    "source_path": "/tmp/alpha.pdf",
                    "content_metadata": {"page_number": 1, "subtype": "table"},
                    "table_metadata": {"table_content": "| A |\n| --- |\n| 1 |"},
                },
            },
            {
                "document_type": "image",
                "metadata": {
                    "source_path": "/tmp/beta.pdf",
                    "content_metadata": {"page_number": 2, "subtype": "page_image"},
                    "image_metadata": {"text": "OCR fallback"},
                },
            },
        ]
    )

    pages = to_markdown_by_page(rows)

    assert "Page text" in pages["alpha.pdf"][1]
    assert "### Table 1" in pages["alpha.pdf"][1]
    assert "### Page Image 1" in pages["beta.pdf"][2]
    assert "OCR fallback" in pages["beta.pdf"][2]


def test_to_markdown_reads_saved_records_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    payload = {
        "source_path": "/tmp/example.pdf",
        "records": [
            {
                "page_number": 1,
                "text": "Saved result text",
                "table": [{"text": "| H |\n| --- |\n| V |"}],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    markdown = to_markdown(path)

    assert list(markdown) == ["example.pdf"]
    assert "Saved result text" in markdown["example.pdf"]
    assert "### Table 1" in markdown["example.pdf"]


def test_to_markdown_empty_results_returns_empty_dict() -> None:
    assert to_markdown([]) == {}


def test_to_markdown_groups_inprocess_multi_document_results() -> None:
    doc_a = pd.DataFrame([{"page_number": 1, "text": "A"}])
    doc_a["path"] = "/tmp/a.pdf"
    doc_b = pd.DataFrame([{"page_number": 1, "text": "B"}])
    doc_b["path"] = "/tmp/b.pdf"

    markdown = to_markdown([doc_a, doc_b])

    assert set(markdown) == {"a.pdf", "b.pdf"}
    assert "A" in markdown["a.pdf"]
    assert "B" in markdown["b.pdf"]


def test_to_markdown_supports_batch_dataset_like_results() -> None:
    results = _BatchResults(
        [
            {
                "document_type": "text",
                "metadata": {
                    "source_path": "/tmp/batch-a.pdf",
                    "content": "Batch A page 1",
                    "content_metadata": {"page_number": 1},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "source_path": "/tmp/batch-b.pdf",
                    "content": "Batch B page 2",
                    "content_metadata": {"page_number": 2},
                },
            },
        ]
    )

    pages = to_markdown_by_page(results)

    assert pages["batch-a.pdf"][1] == "Batch A page 1"
    assert pages["batch-b.pdf"][2] == "Batch B page 2"
