import sys
from pathlib import Path

import pytest

from nemo_retriever.tools.recall.beir import (
    BeirConfig,
    BeirDataset,
    BO767_ANNOTATIONS_PATH,
    DEFAULT_BEIR_KS,
    build_beir_run_from_hits,
    build_qrels_by_query_id,
    build_queries_by_id,
    compute_beir_metrics,
    compute_beir_breakdowns,
    evaluate_lancedb_beir,
    load_beir_dataset,
    resolve_beir_dataset_options,
)


def test_build_queries_by_id_filters_language() -> None:
    rows = [
        {"query_id": 1, "query": "what is a qubit?", "language": "english"},
        {"query_id": 2, "query": "bonjour", "language": "french"},
    ]

    query_ids, queries = build_queries_by_id(rows, query_language="english")

    assert query_ids == ["1"]
    assert queries == ["what is a qubit?"]


def test_build_queries_by_id_filters_language_aliases() -> None:
    rows = [
        {"query_id": 1, "query": "bonjour", "language": "Français"},
        {"query_id": 2, "query": "salut", "language": "french"},
        {"query_id": 3, "query": "hello", "language": "en"},
    ]

    query_ids, queries = build_queries_by_id(rows, query_language="fr")

    assert query_ids == ["1", "2"]
    assert queries == ["bonjour", "salut"]


def test_build_queries_by_id_filters_non_english_language_aliases() -> None:
    rows = [
        {"query_id": 1, "query": "guten tag", "language": "german"},
        {"query_id": 2, "query": "hallo", "language": "Deutsch"},
        {"query_id": 3, "query": "hola", "language": "español"},
        {"query_id": 4, "query": "こんにちは", "language": "japanese"},
    ]

    query_ids, queries = build_queries_by_id(rows, query_language="de")

    assert query_ids == ["1", "2"]
    assert queries == ["guten tag", "hallo"]


def test_build_queries_by_id_warns_when_all_queries_filtered(caplog) -> None:
    rows = [
        {"query_id": 1, "query": "", "language": "en"},
        {"query_id": 2, "query": "bonjour", "language": "fr"},
    ]

    with caplog.at_level("WARNING", logger="nemo_retriever.tools.recall.beir"):
        query_ids, queries = build_queries_by_id(rows, query_language="en")

    assert query_ids == []
    assert queries == []
    assert "No BEIR queries loaded from rows" in caplog.text
    assert "total=2" in caplog.text
    assert "skipped_empty=1" in caplog.text
    assert "skipped_language=1" in caplog.text


def test_build_queries_by_id_warning_logs_normalized_query_language(caplog) -> None:
    rows = [{"query_id": 1, "query": "hello", "language": "english"}]

    with caplog.at_level("WARNING", logger="nemo_retriever.tools.recall.beir"):
        query_ids, queries = build_queries_by_id(rows, query_language="Français")

    assert query_ids == []
    assert queries == []
    assert "query_language='fr'" in caplog.text


def test_build_qrels_by_query_id_formats_nested_dict() -> None:
    rows = [
        {"query_id": 1, "corpus_id": "doc_a", "score": 1},
        {"query_id": 1, "corpus_id": "doc_b", "score": 2},
        {"query_id": 2, "corpus_id": "doc_c", "score": 1},
    ]

    qrels = build_qrels_by_query_id(rows, allowed_query_ids={"1"})

    assert qrels == {"1": {"doc_a": 1, "doc_b": 2}}


def test_build_beir_run_from_hits_uses_pdf_basename_and_dedupes() -> None:
    raw_hits = [
        [
            {"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"},
            {"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"},
            {"pdf_basename": "doc_b", "source_id": "/tmp/doc_b.pdf"},
        ]
    ]

    run = build_beir_run_from_hits(["q1"], raw_hits, doc_id_field="pdf_basename")

    assert list(run["q1"].keys()) == ["doc_a", "doc_b"]
    assert run["q1"]["doc_a"] > run["q1"]["doc_b"]


def test_resolve_beir_dataset_options_supports_known_dataset_name() -> None:
    options = resolve_beir_dataset_options(dataset_name="bo767")

    assert options.loader == "bo767_csv"
    assert options.dataset_name == str(BO767_ANNOTATIONS_PATH)
    assert options.doc_id_field == "pdf_page"
    assert options.ks == DEFAULT_BEIR_KS


def test_resolve_beir_dataset_options_supports_vidore_dataset_name() -> None:
    options = resolve_beir_dataset_options(dataset_name="vidore_v3_computer_science")

    assert options.loader == "vidore_hf"
    assert options.dataset_name == "vidore_v3_computer_science"
    assert options.doc_id_field == "pdf_basename"


def test_resolve_beir_dataset_options_preserves_explicit_overrides(tmp_path: Path) -> None:
    annotations = tmp_path / "custom.csv"

    options = resolve_beir_dataset_options(
        dataset_name=str(annotations),
        loader="bo767_csv",
        doc_id_field="pdf_page_modality",
        ks=[5],
    )

    assert options.loader == "bo767_csv"
    assert options.dataset_name == str(annotations)
    assert options.doc_id_field == "pdf_page_modality"
    assert options.ks == (5,)


def test_resolve_beir_dataset_options_does_not_guess_unknown_dataset() -> None:
    options = resolve_beir_dataset_options(dataset_name="custom_dataset")

    assert options.loader is None
    assert options.dataset_name == "custom_dataset"
    assert options.doc_id_field == "pdf_basename"
    assert options.ks == DEFAULT_BEIR_KS


def test_load_beir_dataset_supports_bo767_csv_pdf_page_modality(tmp_path: Path) -> None:
    annotations = tmp_path / "bo767_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo767_csv", dataset_name=str(annotations), doc_id_field="pdf_page_modality")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1_text": 1},
        "1": {"1002_5_table": 1},
    }


def test_load_beir_dataset_supports_bo767_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "bo767_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo767_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
    }


def test_load_beir_dataset_supports_bo10k_csv_pdf_page_modality(tmp_path: Path) -> None:
    annotations = tmp_path / "digital_corpora_10k_annotations.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("bo10k_csv", dataset_name=str(annotations), doc_id_field="pdf_page_modality")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1_text": 1},
        "1": {"1002_5_table": 1},
    }


def test_load_beir_dataset_supports_earnings_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is doc a?,Answer A,1001,0",
                "table,What is doc b?,Answer B,1002.pdf,4",
                "chart,What is doc c?,Answer C,1003.pdf,2",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1", "2"]
    assert dataset.queries == ["What is doc a?", "What is doc b?", "What is doc c?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
        "2": {"1003_3": 1},
    }


def test_load_beir_dataset_supports_jp20_csv_pdf_page(tmp_path: Path) -> None:
    annotations = tmp_path / "jp20_query_gt.csv"
    annotations.write_text(
        "\n".join(
            [
                "query,pdf,page,pdf_page",
                "What is doc a?,1001,0,1001_1",
                "What is doc b?,1002.pdf,4,1002_5",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("jp20_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.query_ids == ["0", "1"]
    assert dataset.queries == ["What is doc a?", "What is doc b?"]
    assert dataset.qrels == {
        "0": {"1001_1": 1},
        "1": {"1002_5": 1},
    }


def test_load_beir_dataset_supports_financebench_json_pdf_basename(tmp_path: Path) -> None:
    annotations = tmp_path / "financebench_train.json"
    annotations.write_text(
        '[{"id":"q1","question":"What is revenue?","contexts":[{"filename":"AAPL_2023.pdf"}]},'
        '{"id":"q2","question":" What is margin? ","contexts":[{"filename":"MSFT_2022"}]}]',
        encoding="utf-8",
    )

    dataset = load_beir_dataset("financebench_json", dataset_name=str(annotations), doc_id_field="pdf_basename")

    assert dataset.query_ids == ["q1", "q2"]
    assert dataset.queries == ["What is revenue?", " What is margin? "]
    assert dataset.qrels == {
        "q1": {"AAPL_2023": 1},
        "q2": {"MSFT_2022": 1},
    }


def test_load_beir_dataset_preserves_dotted_pdf_basenames_without_extension(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is fair value?,Answer A,3.-Facebook-Reports-Third-Quarter-2016-Results,0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.qrels == {"0": {"3.-Facebook-Reports-Third-Quarter-2016-Results_1": 1}}


def test_load_beir_dataset_preserves_query_whitespace_for_retrieval_parity(tmp_path: Path) -> None:
    annotations = tmp_path / "earnings_consulting_multimodal.csv"
    annotations.write_text(
        "\n".join(
            [
                "modality,query,answer,pdf,page",
                "text,What is the Apple arcade? ,Answer A,1001,0",
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_beir_dataset("earnings_csv", dataset_name=str(annotations), doc_id_field="pdf_page")

    assert dataset.queries == ["What is the Apple arcade? "]


def test_build_beir_run_from_hits_synthesizes_pdf_page_modality() -> None:
    raw_hits = [
        [
            {
                "source": '{"source_id": "/tmp/doc_a.pdf"}',
                "page_number": 7,
                "metadata": "{'_content_type': 'table_caption'}",
            },
            {
                "source_id": "/tmp/doc_a.pdf",
                "page_number": 7,
                "metadata": '{"content_metadata": {"type": "table"}}',
            },
            {
                "source_id": "/tmp/doc_b.pdf",
                "page_number": 3,
                "metadata": "{'content_metadata': {'type': 'text'}}",
            },
        ]
    ]

    run = build_beir_run_from_hits(["q1"], raw_hits, doc_id_field="pdf_page_modality")

    assert list(run["q1"].keys()) == ["doc_a_7_table", "doc_b_3_text"]
    assert run["q1"]["doc_a_7_table"] > run["q1"]["doc_b_3_text"]


def test_compute_beir_metrics_returns_expected_cutoffs() -> None:
    qrels = {
        "q1": {"doc_a": 1},
        "q2": {"doc_b": 1},
    }
    run = {
        "q1": {"doc_a": 2.0, "doc_c": 1.0},
        "q2": {"doc_c": 2.0, "doc_b": 1.0},
    }

    metrics = compute_beir_metrics(qrels, run, ks=(1, 2))

    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["ndcg@1"] == 0.5


def test_map_cut_uses_all_relevant_targets_and_precision_uses_cutoff() -> None:
    # TREC map_cut: (1/1 + 2/3) / 4, not division by min(3, 4).
    qrels = {"q": {"a": 2, "b": 1, "c": 1, "d": 1}}
    run = {"q": {"a": 4.0, "miss": 3.0, "b": 2.0}}
    metrics = compute_beir_metrics(qrels, run, ks=(1, 3, 10, 100))
    assert metrics["map@1"] == 0.25
    assert metrics["map@3"] == pytest.approx((1 + 2 / 3) / 4)
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["precision@10"] == 0.2
    assert metrics["precision@100"] == 0.02
    assert metrics["recall@3"] == 0.5
    assert metrics["map@100"] == metrics["map@3"]


def test_missing_queries_zero_relevance_and_ties_are_deterministic() -> None:
    qrels = {"hit": {"a": 1}, "missing": {"b": 1}, "zero": {"c": 0}}
    run = {"hit": {"z": 1.0, "a": 1.0}, "extra": {"a": 9.0}}
    metrics = compute_beir_metrics(qrels, run, ks=(1, 5))
    for family in ("map", "precision", "recall", "ndcg"):
        assert metrics[f"{family}@1"] == pytest.approx(1 / 3)
    assert metrics["precision@5"] == pytest.approx(1 / 15)
    assert all(value == 0 for value in compute_beir_metrics(qrels, {}, ks=(1, 100)).values())


def test_breakdowns_filter_query_population_and_report_overlap_and_coverage() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}, "q3": {"c": 1}}
    run = {"q1": {"a": 1.0}, "q3": {"c": 1.0}}
    metadata = {"q1": {"content_type": ["Text", "Table", "Text"], "language": "en"},
                "q2": {"content_type": ["Table"]}, "extra": {"content_type": ["Table"]}}
    result = compute_beir_breakdowns(qrels, run, metadata, ks=(1, 10), fields=("content_type", "language"))
    field = result["fields"]["content_type"]
    assert result["query_count"] == 3
    assert field["covered_query_count"] == 2
    assert field["missing_query_count"] == 1
    assert field["groups"]["Text"]["query_count"] == 1
    assert field["groups"]["Table"]["query_count"] == 2
    assert field["groups"]["Table"]["metrics"] == compute_beir_metrics(
        {key: qrels[key] for key in ("q1", "q2")}, {"q1": run["q1"]}, ks=(1, 10),
    )
    assert result["fields"]["language"]["groups"]["en"]["metrics"]["map@1"] == 1
    assert compute_beir_breakdowns(qrels, run, {}, fields=("modality",))["fields"]["modality"][
        "missing_query_count"] == 3


def test_csv_metadata_does_not_infer_missing_modality(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    path.write_text("query_id,query,pdf,page,modality\nkept,question,doc,0,chart\nmissing,other,doc,1,\n")
    dataset = load_beir_dataset("bo767_csv", dataset_name=str(path), doc_id_field="pdf_page")
    assert dataset.query_metadata == {"kept": {"modality": "chart"}, "missing": {}}
    assert BeirDataset("legacy", [], [], {}).query_metadata == {}


def test_vidore_metadata_follows_language_and_qrel_filtering(monkeypatch) -> None:
    rows = {
        "queries": [{"query_id": 1, "query": "English", "language": "en", "content_type": ["Text"]},
                    {"query_id": 2, "query": "French", "language": "fr", "query_format": "keyword"},
                    {"query_id": 3, "query": "Unlabeled", "language": "en"}],
        "qrels": [{"query_id": 1, "corpus_id": "c", "score": 2},
                  {"query_id": 2, "corpus_id": "c", "score": 1}],
        "corpus": [{"corpus_id": "c", "doc_id": "doc", "page_number_in_doc": 0}],
    }
    def load(_repo, config, **_kwargs):
        return rows[config]
    monkeypatch.setitem(sys.modules, "datasets", type("Datasets", (), {"load_dataset": load}))
    dataset = load_beir_dataset("vidore_hf", dataset_name="test", query_language="en", doc_id_field="pdf_page")
    assert dataset.query_ids == ["1"]
    assert dataset.query_metadata == {"1": {"content_type": ["Text"], "language": "en"}}
    assert dataset.qrels == {"1": {"doc_1": 2}}


def test_load_beir_dataset_tries_vidore_config_name_before_data_dir(monkeypatch) -> None:
    calls = []

    def _fake_load_dataset(repo, *args, **kwargs):
        calls.append((repo, args, kwargs))
        if args == ("queries",):
            return [{"query_id": "q1", "query": "What is shown?", "language": "en"}]
        if args == ("qrels",):
            return [{"query_id": "q1", "corpus_id": "doc_a", "score": 1}]
        if args == ("corpus",):
            return [{"corpus_id": "doc_a", "doc_id": "doc_a"}]
        raise AssertionError("data_dir fallback should not be used")

    monkeypatch.setitem(sys.modules, "datasets", type("Datasets", (), {"load_dataset": _fake_load_dataset}))

    dataset = load_beir_dataset("vidore_hf", dataset_name="vidore_v3_computer_science")

    assert dataset.query_ids == ["q1"]
    assert dataset.qrels == {"q1": {"doc_a": 1}}
    assert calls[0] == ("vidore/vidore_v3_computer_science", ("queries",), {"split": "test"})
    assert calls[1] == ("vidore/vidore_v3_computer_science", ("qrels",), {"split": "test"})
    assert calls[2] == ("vidore/vidore_v3_computer_science", ("corpus",), {"split": "test"})


def test_load_beir_dataset_falls_back_to_vidore_data_dir(monkeypatch) -> None:
    calls = []

    def _fake_load_dataset(repo, *args, **kwargs):
        calls.append((repo, args, kwargs))
        if args:
            raise RuntimeError("config-name unavailable")
        if kwargs.get("data_dir") == "queries":
            return [{"query_id": "q1", "query": "What is shown?", "language": "en"}]
        if kwargs.get("data_dir") == "qrels":
            return [{"query_id": "q1", "corpus_id": "doc_a", "score": 1}]
        if kwargs.get("data_dir") == "corpus":
            return [{"corpus_id": "doc_a", "doc_id": "doc_a"}]
        raise AssertionError("unexpected load_dataset call")

    monkeypatch.setitem(sys.modules, "datasets", type("Datasets", (), {"load_dataset": _fake_load_dataset}))

    dataset = load_beir_dataset("vidore_hf", dataset_name="vidore_v3_computer_science")

    assert dataset.query_ids == ["q1"]
    assert dataset.qrels == {"q1": {"doc_a": 1}}
    assert calls == [
        ("vidore/vidore_v3_computer_science", ("queries",), {"split": "test"}),
        ("vidore/vidore_v3_computer_science", (), {"data_dir": "queries", "split": "test"}),
        ("vidore/vidore_v3_computer_science", ("qrels",), {"split": "test"}),
        ("vidore/vidore_v3_computer_science", (), {"data_dir": "qrels", "split": "test"}),
        ("vidore/vidore_v3_computer_science", ("corpus",), {"split": "test"}),
        ("vidore/vidore_v3_computer_science", (), {"data_dir": "corpus", "split": "test"}),
    ]


def test_load_beir_dataset_error_includes_query_language_and_raw_row_count(monkeypatch) -> None:
    def _fake_load_dataset(_repo, *args, **_kwargs):
        if args == ("queries",):
            return [{"query_id": "q1", "query": "bonjour", "language": "fr"}]
        if args == ("qrels",):
            return [{"query_id": "q1", "corpus_id": "doc_a", "score": 1}]
        raise AssertionError("unexpected load_dataset call")

    monkeypatch.setitem(sys.modules, "datasets", type("Datasets", (), {"load_dataset": _fake_load_dataset}))

    with pytest.raises(ValueError) as exc_info:
        load_beir_dataset("vidore_hf", dataset_name="vidore_v3_computer_science", query_language="en")

    message = str(exc_info.value)
    assert "query_language='en'" in message
    assert "Loaded 1 raw rows from HuggingFace" in message


def test_evaluate_lancedb_beir_uses_loader_and_retriever(monkeypatch) -> None:
    dataset = BeirDataset(
        dataset_name="vidore_v3_computer_science",
        query_ids=["1"],
        queries=["what is a qubit?"],
        qrels={"1": {"doc_a": 1}},
    )

    monkeypatch.setattr(
        "nemo_retriever.tools.recall.beir.load_beir_dataset",
        lambda *args, **kwargs: dataset,
    )

    retriever_instances: list = []

    class _FakeRetriever:
        def __init__(self, **kwargs):
            expected_kwargs = {
                "vdb_kwargs": {
                    "vdb_op": "lancedb",
                    "vdb_kwargs": {
                        "uri": "/tmp/lancedb",
                        "table_name": "nemo-retriever",
                        "hybrid": False,
                        "nprobes": 0,
                        "refine_factor": 10,
                    },
                },
                "embed_kwargs": {
                    "model_name": "embedder",
                    "embed_model_name": "embedder",
                    "local_ingest_embed_backend": "vllm",
                    "inference_batch_size": 32,
                    "embed_inference_batch_size": 32,
                    "query_max_length": 128,
                    "embedding_endpoint": "http://embed.example/v1",
                    "embed_invoke_url": "http://embed.example/v1",
                    "api_key": "secret",
                },
                "top_k": 10,
                "rerank": False,
                "rerank_kwargs": {
                    "model_name": "nvidia/llama-nemotron-rerank-1b-v2",
                    "rerank_invoke_url": None,
                    "api_key": "",
                    "batch_size": 32,
                    "local_reranker_backend": "vllm",
                },
            }
            missing_keys = set(expected_kwargs) - set(kwargs)
            assert not missing_keys
            assert {key: kwargs[key] for key in expected_kwargs} == expected_kwargs
            self.kwargs = kwargs
            retriever_instances.append(self)

        def queries(self, queries):
            assert queries == ["what is a qubit?"]
            return [[{"pdf_basename": "doc_a", "source_id": "/tmp/doc_a.pdf"}]]

    monkeypatch.setattr("nemo_retriever.tools.recall.beir.Retriever", _FakeRetriever)

    cfg = BeirConfig(
        lancedb_uri="/tmp/lancedb",
        lancedb_table="nemo-retriever",
        embedding_model="embedder",
        embedding_http_endpoint="http://embed.example/v1",
        embedding_api_key=" secret ",
        loader="vidore_hf",
        dataset_name="vidore_v3_computer_science",
    )

    loaded_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(cfg)

    assert loaded_dataset == dataset
    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert "embed_use_vllm" not in retriever_instances[0].kwargs
    assert retriever_instances[0].kwargs["embed_kwargs"].get("local_ingest_embed_backend") == "vllm"
    assert retriever_instances[0].kwargs["rerank_kwargs"].get("local_reranker_backend") == "vllm"
