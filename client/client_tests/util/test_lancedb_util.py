import types
from unittest.mock import Mock

from nv_ingest_client.util.vdb import lancedb as lancedb_mod


def make_sample_results():
    # two record-sets, each with two items
    return [
        [
            {
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                }
            },
            {
                "metadata": {
                    "embedding": None,
                    "content": "skip",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                }
            },
        ],
        [
            {
                "metadata": {
                    "embedding": [0.3, 0.4],
                    "content": "b",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                }
            },
        ],
    ]


def test_create_lancedb_results_filters_and_transforms():
    results = make_sample_results()
    out = lancedb_mod.create_lancedb_results(results)
    # Should skip the entry with None embedding and produce 2 entries
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["vector"] == [0.1, 0.2]
    assert out[0]["text"] == "a"
    assert out[0]["metadata"] == 1
    assert out[0]["source"] == "s1"


def test_create_index_uses_lancedb_and_pa_schema(monkeypatch):
    # Mock lancedb.connect and pa.schema so we don't need real packages
    fake_db = Mock()
    fake_table = Mock()
    fake_db.create_table = Mock(return_value=fake_table)

    monkeypatch.setattr(lancedb_mod, "lancedb", Mock(connect=Mock(return_value=fake_db)))
    monkeypatch.setattr(lancedb_mod, "pa", Mock(schema=Mock(return_value="schema")))

    sample = make_sample_results()
    # call create_index with the same nested input shape the function expects
    table = lancedb_mod.LanceDB().create_index(records=sample)

    # Ensure create_table was called and returned the fake table
    fake_db.create_table.assert_called()
    assert table is fake_table


def test_write_to_index_creates_index_and_waits(monkeypatch):
    # Create a fake table object with expected methods
    fake_table = Mock()
    fake_table.list_indices = Mock(
        return_value=[types.SimpleNamespace(name="idx1"), types.SimpleNamespace(name="idx2")]
    )
    fake_table.create_index = Mock()
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB()
    # write_to_index uses table.create_index then waits for indices
    ldb.write_to_index(records=None, table=fake_table)

    fake_table.create_index.assert_called()
    fake_table.list_indices.assert_called()
    # Ensure wait_for_index was called with list of names
    fake_table.wait_for_index.assert_called()


def test_retrieval_calls_embed_and_search(monkeypatch):
    # Prepare a fake infer_microservice that returns one embedding per query
    def fake_infer_microservice(queries, **kwargs):
        # return a list of embeddings (one per query)
        return [[0.1, 0.2, 0.3] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    # Build a fake table.search chain: search(...).select(...).limit(...).to_list()
    class FakeSearchResult:
        def __init__(self, items):
            self._items = items

        def select(self, cols):
            return self

        def limit(self, n):
            return self

        def to_list(self):
            return self._items

    fake_table = Mock()
    # Return two hits for each search to verify wrapping
    fake_table.search = Mock(return_value=FakeSearchResult([{"text": "hit1", "metadata": 1, "source": "s"}]))

    ldb = lancedb_mod.LanceDB()
    queries = ["hello", "world"]
    results = ldb.retrieval(queries, table=fake_table)

    # Should return a list with an entry per query
    assert isinstance(results, list)
    assert len(results) == len(queries)
    # Each entry should be the list returned by FakeSearchResult.to_list()
    assert results[0][0]["text"] == "hit1"


def test_run_calls_create_and_write(monkeypatch):
    ldb = lancedb_mod.LanceDB()
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
    )
    assert records == out


# ============ Additional Tests for Untested Areas ============


def test_init_with_uri_and_overwrite():
    """Test LanceDB initialization with custom uri and overwrite settings."""
    ldb = lancedb_mod.LanceDB(uri="custom_uri", overwrite=False, extra_key="value")
    assert ldb.uri == "custom_uri"
    assert ldb.overwrite is False
    assert ldb.extra_key == "value"  # Verify kwargs are stored


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
                "metadata": {
                    "embedding": None,
                    "content": "a",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                }
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
                "metadata": {
                    "embedding": [0.1],
                    "content": "keep",
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_id": "s1"},
                }
            },
            {
                "metadata": {
                    "embedding": None,
                    "content": "skip",
                    "content_metadata": {"page_number": 2},
                    "source_metadata": {"source_id": "s1"},
                }
            },
            {
                "metadata": {
                    "embedding": [0.2, 0.3],
                    "content": "keep2",
                    "content_metadata": {"page_number": 3},
                    "source_metadata": {"source_id": "s2"},
                }
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
    fake_table.wait_for_index = Mock()

    ldb = lancedb_mod.LanceDB()
    custom_kwargs = {
        "index_type": "custom_index",
        "metric": "cosine",
        "num_partitions": 32,
        "num_sub_vectors": 512,
    }
    ldb.write_to_index(records=None, table=fake_table, **custom_kwargs)

    # Verify custom_kwargs were passed to create_index
    call_kwargs = fake_table.create_index.call_args[1]
    assert call_kwargs["index_type"] == "custom_index"
    assert call_kwargs["metric"] == "cosine"
    assert call_kwargs["num_partitions"] == 32
    assert call_kwargs["num_sub_vectors"] == 512


def test_retrieval_with_custom_embedding_params(monkeypatch):
    """Test retrieval with custom embedding endpoint and model."""

    def fake_infer_microservice(queries, **kwargs):
        # Capture kwargs and return embeddings
        fake_infer_microservice.captured_kwargs = kwargs
        return [[0.1, 0.2] for _ in queries]

    monkeypatch.setattr(lancedb_mod, "infer_microservice", fake_infer_microservice)

    class FakeSearchResult:
        def select(self, cols):
            return self

        def limit(self, n):
            return self

        def to_list(self):
            return [{"text": "result", "metadata": 1, "source": "s"}]

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult())

    ldb = lancedb_mod.LanceDB()
    custom_endpoint = "http://custom-endpoint:9000/v1"
    custom_model = "my/custom-model"
    custom_key = "my-api-key"

    ldb.retrieval(
        ["test query"],
        table=fake_table,
        embedding_endpoint=custom_endpoint,
        model_name=custom_model,
        nvidia_api_key=custom_key,
    )

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

        def to_list(self):
            return [{"text": "result"}]

    fake_table = Mock()
    fake_table.search = Mock(return_value=FakeSearchResult())

    ldb = lancedb_mod.LanceDB()
    ldb.retrieval(["query"], table=fake_table)

    # Default top_k is 10
    assert captured_limit[0] == 10


def test_create_lancedb_results_missing_field_raises_keyerror():
    """Test that missing required fields raise KeyError."""
    results = [
        [
            {
                "metadata": {
                    "embedding": [0.1],
                    "content": "a",
                    # Missing content_metadata
                    "source_metadata": {"source_id": "s1"},
                }
            }
        ]
    ]
    try:
        lancedb_mod.create_lancedb_results(results)
        assert False, "Expected KeyError for missing content_metadata"
    except KeyError:
        pass  # Expected


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
    assert results[0][0]["text"] == "result_0"
    assert results[1][0]["text"] == "result_1"
    assert results[2][0]["text"] == "result_2"
