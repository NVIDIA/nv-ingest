import pytest

pytest.importorskip("ray")

from nemo_retriever.examples.batch_pipeline import _count_materialized_rows
from nemo_retriever.examples.batch_pipeline import _resolve_input_file_patterns


class _DatasetWithoutLen:
    def count(self) -> int:
        return 42

    def __len__(self) -> int:
        raise AssertionError("__len__ should not be used")


def test_count_materialized_rows_prefers_dataset_count() -> None:
    assert _count_materialized_rows(_DatasetWithoutLen()) == 42


def test_resolve_input_file_patterns_recurses_for_directory_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    dataset_dir.mkdir()

    pdf_patterns = _resolve_input_file_patterns(dataset_dir, "pdf")
    txt_patterns = _resolve_input_file_patterns(dataset_dir, "txt")
    doc_patterns = _resolve_input_file_patterns(dataset_dir, "doc")

    assert pdf_patterns == [str(dataset_dir / "**" / "*.pdf")]
    assert txt_patterns == [str(dataset_dir / "**" / "*.txt")]
    assert doc_patterns == [str(dataset_dir / "**" / "*.docx"), str(dataset_dir / "**" / "*.pptx")]
