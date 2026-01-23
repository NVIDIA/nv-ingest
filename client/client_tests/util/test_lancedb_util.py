import types
from unittest.mock import Mock

from nv_ingest_client.util.vdb import lancedb as lancedb_mod


def make_sample_results():
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
                    "embedding": None,
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
    results = make_sample_results()
    out = lancedb_mod.create_lancedb_results(results)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["vector"] == [0.1, 0.2]
    assert out[0]["text"] == "text content a"
    assert out[0]["metadata"] == "1"
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


def test_get_text_for_element_text_type():
    element = {
        "document_type": "text",
        "metadata": {"content": "This is plain text content"},
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "This is plain text content"


def test_get_text_for_element_page_image_uses_text():
    element = {
        "document_type": "image",
        "metadata": {
            "content_metadata": {"subtype": "page_image"},
            "image_metadata": {"text": "OCR extracted text from page", "caption": "Should not use this"},
        },
    }
    result = lancedb_mod._get_text_for_element(element)
    assert result == "OCR extracted text from page"


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
        return [[0.1, 0.2, 0.3] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

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
