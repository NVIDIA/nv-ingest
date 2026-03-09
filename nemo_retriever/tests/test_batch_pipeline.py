import pytest

pytest.importorskip("ray")

from nemo_retriever.examples.batch_pipeline import _resolve_input_file_patterns


def test_resolve_input_file_patterns_recurses_for_directory_inputs(tmp_path) -> None:
    dataset_dir = tmp_path / "earnings_consulting"
    dataset_dir.mkdir()

    pdf_patterns = _resolve_input_file_patterns(dataset_dir, "pdf")
    txt_patterns = _resolve_input_file_patterns(dataset_dir, "txt")
    doc_patterns = _resolve_input_file_patterns(dataset_dir, "doc")

    assert pdf_patterns == [str(dataset_dir / "**" / "*.pdf")]
    assert txt_patterns == [str(dataset_dir / "**" / "*.txt")]
    assert doc_patterns == [str(dataset_dir / "**" / "*.docx"), str(dataset_dir / "**" / "*.pptx")]
