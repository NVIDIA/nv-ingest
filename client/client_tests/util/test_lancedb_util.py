import sys
import types
from unittest.mock import Mock

# Ensure the client src path is importable
sys.path.insert(0, "/raid/jperez/nv-ingest/client/src")

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

    ldb.create_index.assert_called_once_with(records=records)
    ldb.write_to_index.assert_called_once_with(records, table="table_obj")
    assert out == records
