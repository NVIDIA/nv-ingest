import json

import pytest
from nv_ingest_client.client.util.processing import save_document_results_to_jsonl


@pytest.fixture
def sample_doc_response_data():
    return [
        {"id": 1, "type": "text", "content": "Hello world"},
        {"id": 2, "type": "image", "content": "base64_image_data_here"},
        {"id": 3, "type": "text", "content": "Another line"},
    ]


def test_save_document_results_to_jsonl_successful_save(tmp_path, sample_doc_response_data, caplog):
    output_filename = "doc1.results.jsonl"
    jsonl_filepath = tmp_path / output_filename
    source_name = "document_source_1.pdf"

    count_saved = save_document_results_to_jsonl(sample_doc_response_data, str(jsonl_filepath), source_name)

    assert count_saved == len(sample_doc_response_data)
    assert jsonl_filepath.exists()

    loaded_data = []
    with open(jsonl_filepath, "r", encoding="utf-8") as f:
        for line in f:
            loaded_data.append(json.loads(line))
    assert loaded_data == sample_doc_response_data
