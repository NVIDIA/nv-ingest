from nemo_retriever.harness.parsers import StreamMetrics, parse_stream_text


def test_parse_stream_text_extracts_done_recall_and_pages_per_second() -> None:
    stdout = """
[done] 20 files, 3181 pages in 122.3s
Recall metrics (matching nemo_retriever.recall.core):
  recall@1: 0.6087
  recall@5: 0.9043
  recall@10: 0.9565
Pages/sec (ingest only; excludes Ray startup and recall): 15.77
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files == 20
    assert metrics.pages == 3181
    assert metrics.ingest_secs == 122.3
    assert metrics.pages_per_sec_ingest == 15.77
    assert metrics.recall_metrics == {
        "recall@1": 0.6087,
        "recall@5": 0.9043,
        "recall@10": 0.9565,
    }
    assert metrics.rows_processed is None
    assert metrics.rows_per_sec_ingest is None


def test_parse_stream_text_extracts_rows_and_logger_prefixed_recall_lines() -> None:
    stdout = """
2026-03-06 20:12:42,120 INFO nemo_retriever.examples.batch_pipeline: Ingestion complete. 3181 rows procesed in 151.40 seconds. 21.01 PPS
2026-03-06 20:12:42,121 INFO nemo_retriever.examples.batch_pipeline: Recall metrics (matching nemo_retriever.recall.core):
2026-03-06 20:12:42,122 INFO nemo_retriever.examples.batch_pipeline:   recall@1: 0.6087
2026-03-06 20:12:42,123 INFO nemo_retriever.examples.batch_pipeline:   recall@5: 0.9043
2026-03-06 20:12:42,124 INFO nemo_retriever.examples.batch_pipeline:   recall@10: 0.9565
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files is None
    assert metrics.pages is None
    assert metrics.ingest_secs == 151.40
    assert metrics.pages_per_sec_ingest is None
    assert metrics.rows_processed == 3181
    assert metrics.rows_per_sec_ingest == 21.01
    assert metrics.recall_metrics == {
        "recall@1": 0.6087,
        "recall@5": 0.9043,
        "recall@10": 0.9565,
    }


def test_stream_metrics_handles_non_recall_lines_after_recall_block() -> None:
    metrics = StreamMetrics()
    metrics.consume("Recall metrics (matching nemo_retriever.recall.core):\n")
    metrics.consume("  recall@5: 0.9043\n")
    metrics.consume("Pages processed: 1933\n")
    metrics.consume("  recall@10: 0.9565\n")
    assert metrics.recall_metrics == {"recall@5": 0.9043}
