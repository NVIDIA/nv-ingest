import types
from unittest.mock import Mock

from nv_ingest_client.util.vdb import lancedb as lancedb_mod


def make_sample_results():
    """
    Create sample NV-Ingest pipeline results for testing.

    Returns records in the format produced by the extraction pipeline,
    including document_type which is required for text extraction routing.
    """
    return [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "text content a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,  # Should be skipped - no embedding
                    "content": "skip me",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                },
            },
        ],
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.3, 0.4],
                    "content": "text content b",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                },
            },
        ],
    ]


def test_create_lancedb_results_filters_and_transforms():
    """Test that create_lancedb_results correctly filters and transforms records."""
    results = make_sample_results()
    out = lancedb_mod.create_lancedb_results(results)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["vector"] == [0.1, 0.2]
    assert out[0]["text"] == "text content a"
    assert out[0]["metadata"] == "1"  # page_number converted to string
    assert out[0]["source"] == "s1"


def test_create_index_uses_lancedb_and_pa_schema(monkeypatch):
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    sample = make_sample_results()
    table = lancedb_mod.LanceDB().create_index(records=sample)

    fake_db.create_table.assert_called()
    assert table is fake_table


def test_write_to_index_creates_index_and_waits(monkeypatch):
    fake_table = Mock()
    fake_table.list_indices = Mock(
        return_value=[types.SimpleNamespace(name="idx1"), types.SimpleNamespace(name="idx2")]
    )
    fake_table.create_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB()
    ldb.write_to_index(records=None, table=fake_table)

    fake_table.create_index.assert_called()
    fake_table.list_indices.assert_called()
    fake_table.wait_for_index.assert_called()


def test_retrieval_calls_embed_and_search(monkeypatch):
    def fake_infer_microservice(queries, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    class FakeSearchResult:
        def __init__(self, items):
            self._items = items

        def select(self, cols):
            return self

        def limit(self, n):
            return self

        def refine_factor(self, n):
            return self

        def nprobes(self, n):
            return self

        def to_list(self):
            return self._items

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult([{"text": "hit1", "metadata": 1, "source": "s"}]))

    ldb = lancedb_mod.LanceDB()
    queries = ["hello", "world"]
    results = ldb.retrieval(queries, table=fake_table)

    assert isinstance(results, list)
    assert len(results) == len(queries)
    assert results[0][0]["entity"]["text"] == "hit1"


def test_run_calls_create_and_write(monkeypatch):
    ldb = lancedb_mod.LanceDB(hybrid=True, fts_language="Spanish")
    monkeypatch.setattr(ldb, "create_index", Mock(return_value="table_obj"))
    monkeypatch.setattr(ldb, "write_to_index", Mock())

    records = make_sample_results()
    out = ldb.run(records)

    ldb.create_index.assert_called_once_with(records=records, table_name="nv-ingest")
    ldb.write_to_index.assert_called_once_with(
        records,
        table="table_obj",
        index_type=ldb.index_type,
        metric=ldb.metric,
        num_partitions=ldb.num_partitions,
        num_sub_vectors=ldb.num_sub_vectors,
        hybrid=True,
        fts_language="Spanish",
    )
    assert records == out


def test_write_to_index_creates_fts_index_when_hybrid(monkeypatch):
    fake_table = Mock()
    fake_table.list_indices = Mock(
        return_value=[
            types.SimpleNamespace(name="vector_idx"),
            types.SimpleNamespace(name="text_idx"),
        ]
    )
    fake_table.create_index = Mock()
    fake_table.create_fts_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB(hybrid=True)
    ldb.write_to_index(records=None, table=fake_table, hybrid=True)

    fake_table.create_fts_index.assert_called_once_with("text", language=ldb.fts_language)


def test_init_default_uri_and_overwrite():
    """Test LanceDB initialization with default values."""
    ldb = lancedb_mod.LanceDB()
    assert ldb.uri == "lancedb"
    assert ldb.overwrite is True


def test_create_lancedb_results_empty_list():
    """Test create_lancedb_results with empty input."""
    results = []
    out = lancedb_mod.create_lancedb_results(results)
    assert out == []


def test_create_lancedb_results_all_none_embeddings():
    """Test create_lancedb_results when all embeddings are None."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,
                    "content": "a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results)
    assert out == []


def test_create_lancedb_results_mixed_embeddings():
    """Test create_lancedb_results with mixed None and valid embeddings."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1],
                    "content": "keep",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": None,
                    "content": "skip",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.2, 0.3],
                    "content": "keep2",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                },
            },
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results)
    assert len(out) == 2
    assert out[0]["text"] == "keep"
    assert out[1]["text"] == "keep2"


def test_create_index_uses_uri_from_instance(monkeypatch):
    """Test that create_index uses the uri stored on the instance."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    ldb = lancedb_mod.LanceDB(uri="my_custom_uri")
    sample = make_sample_results()
    ldb.create_index(records=sample)

    # Verify lancedb.connect was called with the custom uri
    lancedb_mod.lancedb.connect.assert_called_once_with(uri="my_custom_uri")


def test_create_index_uses_table_name_kwarg(monkeypatch):
    """Test that create_index respects table_name keyword argument."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    ldb = lancedb_mod.LanceDB()
    sample = make_sample_results()
    ldb.create_index(records=sample, table_name="custom_table")

    # Verify create_table was called with custom_table
    call_args = fake_db.create_table.call_args
    assert call_args[0][0] == "custom_table"


def test_create_index_uses_overwrite_mode(monkeypatch):
    """Test that create_index uses mode based on overwrite setting."""
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    # Test with overwrite=False (should use "append" mode)
    ldb = lancedb_mod.LanceDB(overwrite=False)
    sample = make_sample_results()
    ldb.create_index(records=sample)

    call_kwargs = fake_db.create_table.call_args[1]
    assert call_kwargs["mode"] == "append"

    # Reset and test with overwrite=True (should use "overwrite" mode)
    fake_db.reset_mock()
    ldb2 = lancedb_mod.LanceDB(overwrite=True)
    ldb2.create_index(records=sample)

    call_kwargs = fake_db.create_table.call_args[1]
    assert call_kwargs["mode"] == "overwrite"


def test_write_to_index_respects_kwargs(monkeypatch):
    """Test that write_to_index can accept and use custom index parameters."""
    fake_table = Mock()
    fake_table.list_indices = Mock(return_value=[types.SimpleNamespace(name="idx1")])
    fake_table.create_index = Mock()
    fake_table.create_fts_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB(hybrid=True, fts_language="English")
    ldb.write_to_index(records=None, table=fake_table)

    fake_table.create_index.assert_called_once()
    fake_table.create_fts_index.assert_called_once_with("text", language="English")


def test_retrieval_dispatches_to_hybrid(monkeypatch):
    mock_hybrid_retrieval = Mock(return_value=[["hybrid_result"]])
    mock_regular_retrieval = Mock(return_value=[["regular_result"]])

    monkeypatch.setattr(lancedb_mod, "lancedb_hybrid_retrieval", mock_hybrid_retrieval)
    monkeypatch.setattr(lancedb_mod, "lancedb_retrieval", mock_regular_retrieval)

    ldb = lancedb_mod.LanceDB(hybrid=True)
    results = ldb.retrieval(["test query"])

    mock_hybrid_retrieval.assert_called_once()
    mock_regular_retrieval.assert_not_called()
    assert results == [["hybrid_result"]]


def test_hybrid_retrieval_search_chain(monkeypatch):
    def fake_infer_microservice(queries, **kwargs):
        fake_infer_microservice.captured_kwargs = kwargs
        return [[0.1, 0.2, 0.3] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    custom_endpoint = "http://custom-embed:8012/v1"
    custom_model = "custom/embedding-model"
    custom_key = "custom-key"

    captured_calls = {}

    class FakeHybridSearchResult:
        def vector(self, vec):
            captured_calls["vector"] = vec
            return self

        def text(self, txt):
            captured_calls["text"] = txt
            return self

        def select(self, cols):
            return self

        def limit(self, n):
            captured_calls["limit"] = n
            return self

        def refine_factor(self, n):
            captured_calls["refine_factor"] = n
            return self

        def nprobes(self, n):
            captured_calls["nprobes"] = n
            return self

        def rerank(self, reranker):
            captured_calls["reranker"] = reranker
            return self

        def to_list(self):
            return [{"text": "hybrid_hit", "metadata": "1", "source": "s1"}]

    def fake_search(query_type=None):
        captured_calls["query_type"] = query_type
        return FakeHybridSearchResult()

    fake_table = Mock()
    fake_table.search = Mock(side_effect=fake_search)

    results = lancedb_mod.lancedb_hybrid_retrieval(
        queries=["test query"],
        table=fake_table,
        top_k=5,
        refine_factor=25,
        n_probe=32,
        embedding_endpoint=custom_endpoint,
        model_name=custom_model,
        nvidia_api_key=custom_key,
    )

    assert captured_calls["query_type"] == "hybrid"
    assert captured_calls["vector"] == [0.1, 0.2, 0.3]
    assert captured_calls["text"] == "test query"
    assert captured_calls["limit"] == 5
    assert captured_calls["refine_factor"] == 25
    assert captured_calls["nprobes"] == 32
    assert captured_calls["reranker"] is not None
    assert isinstance(captured_calls["reranker"], lancedb_mod.RRFReranker)
    assert len(results) == 1
    assert results[0][0]["entity"]["text"] == "hybrid_hit"
    # Verify custom params were passed to infer_microservice
    assert fake_infer_microservice.captured_kwargs["embedding_endpoint"] == custom_endpoint
    assert fake_infer_microservice.captured_kwargs["model_name"] == custom_model
    assert fake_infer_microservice.captured_kwargs["nvidia_api_key"] == custom_key


def test_retrieval_with_custom_result_fields(monkeypatch):
    """Test retrieval with custom result fields."""

    def fake_infer_microservice(queries, **kwargs):
        return [[0.1] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    captured_select_fields = []

    class FakeSearchResult:
        def __init__(self):
            self.select_fields = None

        def select(self, cols):
            captured_select_fields.append(cols)
            return self

        def limit(self, n):
            return self

        def refine_factor(self, n):
            return self

        def nprobes(self, n):
            return self

        def to_list(self):
            return [{"custom_field": "value"}]

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult())

    ldb = lancedb_mod.LanceDB()
    custom_fields = ["field1", "field2", "field3"]

    ldb.retrieval(
        ["query"],
        table=fake_table,
        result_fields=custom_fields,
    )

    assert captured_select_fields[0] == custom_fields


def test_retrieval_with_custom_top_k(monkeypatch):
    """Test retrieval respects custom top_k parameter."""

    def fake_infer_microservice(queries, **kwargs):
        return [[0.1] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    captured_limit = []

    class FakeSearchResult:
        def select(self, cols):
            return self

        def limit(self, n):
            captured_limit.append(n)
            return self

        def refine_factor(self, n):
            return self

        def nprobes(self, n):
            return self

        def to_list(self):
            return [{"text": "result"}]

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult())

    ldb = lancedb_mod.LanceDB()
    ldb.retrieval(
        ["query"],
        table=fake_table,
        top_k=50,
    )

    assert captured_limit[0] == 50


def test_retrieval_default_top_k(monkeypatch):
    """Test retrieval uses default top_k when not specified."""

    def fake_infer_microservice(queries, **kwargs):
        return [[0.1] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    captured_limit = []

    class FakeSearchResult:
        def select(self, cols):
            return self

        def limit(self, n):
            captured_limit.append(n)
            return self

        def refine_factor(self, n):
            return self

        def nprobes(self, n):
            return self

        def to_list(self):
            return [{"text": "result"}]

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult())

    ldb = lancedb_mod.LanceDB()
    ldb.retrieval(["query"], table=fake_table)

    # Default top_k is 10
    assert captured_limit[0] == 10


def test_create_lancedb_results_missing_content_metadata_graceful():
    """Test that missing content_metadata is handled gracefully (empty dict default)."""
    results = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1],
                    "content": "text with missing content_metadata",
                    # Missing content_metadata - should use empty dict default
                    "source_metadata": {"source_id": "s1"},
                },
            }
        ]
    ]
    out = lancedb_mod.create_lancedb_results(results)
    # Should still produce output with empty string for page number
    assert len(out) == 1
    assert out[0]["metadata"] == ""  # No page_number available
    assert out[0]["text"] == "text with missing content_metadata"


def test_retrieval_multiple_queries(monkeypatch):
    """Test retrieval processes multiple queries correctly."""

    def fake_infer_microservice(queries, **kwargs):
        return [[0.1 * i] for i in range(len(queries))]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    class FakeSearchResult:
        def __init__(self, query_idx):
            self.query_idx = query_idx

        def select(self, cols):
            return self

        def limit(self, n):
            return self

        def refine_factor(self, n):
            return self

        def nprobes(self, n):
            return self

        def to_list(self):
            return [{"text": f"result_{self.query_idx}"}]

    search_call_count = [0]

    def make_search_result(query_vec, **kwargs):
        result = FakeSearchResult(search_call_count[0])
        search_call_count[0] += 1
        return result

    fake_table = Mock()
    fake_table.search = Mock(side_effect=make_search_result)

    ldb = lancedb_mod.LanceDB()
    queries = ["query1", "query2", "query3"]
    results = ldb.retrieval(queries, table=fake_table)

    # Should have results for all 3 queries
    assert len(results) == 3
    assert results[0][0]["entity"]["text"] == "result_0"
    assert results[1][0]["entity"]["text"] == "result_1"
    assert results[2][0]["entity"]["text"] == "result_2"


# ============ Tests for _get_text_for_element() ============


def test_get_text_for_element_text_type():
    """Test text extraction for document_type='text'."""
    element = {
        "document_type": "text",
        "metadata": {
            "content": "This is plain text content",
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "This is plain text content"


def test_get_text_for_element_structured_table():
    """Test text extraction for structured tables."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "table"},
            "table_metadata": {"table_content": "| Col1 | Col2 |\n| A | B |"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "| Col1 | Col2 |\n| A | B |"


def test_get_text_for_element_structured_chart():
    """Test text extraction for structured charts."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "chart"},
            "table_metadata": {"table_content": "Chart showing revenue trends"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Chart showing revenue trends"


def test_get_text_for_element_structured_infographic():
    """Test text extraction for infographics."""
    element = {
        "document_type": "structured",
        "metadata": {
            "content_metadata": {"subtype": "infographic"},
            "table_metadata": {"table_content": "Infographic content description"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Infographic content description"


def test_get_text_for_element_image_caption():
    """Test text extraction for images uses caption (not base64 data)."""
    element = {
        "document_type": "image",
        "metadata": {
            "content_metadata": {"subtype": "image"},
            "image_metadata": {"caption": "A photo of a dog playing"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "A photo of a dog playing"


def test_get_text_for_element_page_image_uses_text():
    """Test that page_image subtype uses 'text' field (OCR) instead of caption."""
    element = {
        "document_type": "image",
        "metadata": {
            "content_metadata": {"subtype": "page_image"},
            "image_metadata": {
                "text": "OCR extracted text from page",
                "caption": "Should not use this",
            },
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "OCR extracted text from page"


def test_get_text_for_element_audio():
    """Test text extraction for audio uses transcript."""
    element = {
        "document_type": "audio",
        "metadata": {
            "audio_metadata": {"audio_transcript": "Hello, this is the audio transcript"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Hello, this is the audio transcript"


def test_get_text_for_element_unknown_type_fallback():
    """Test that unknown document types fall back to metadata.content."""
    element = {
        "document_type": "unknown_type",
        "metadata": {
            "content": "Fallback content",
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "Fallback content"


def test_get_text_for_element_missing_metadata():
    """Test graceful handling of missing metadata fields."""
    element = {
        "document_type": "text",
        "metadata": {},  # No content field
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result is None
