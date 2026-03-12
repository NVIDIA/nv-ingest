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
2026-03-06 20:12:42,120 INFO nemo_retriever.examples.batch_pipeline: \
Ingestion complete. 3181 rows procesed in 151.40 seconds. 21.01 PPS
2026-03-06 20:12:42,121 INFO nemo_retriever.examples.batch_pipeline: \
Recall metrics (matching nemo_retriever.recall.core):
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


def test_parse_stream_text_extracts_current_run_summary_format() -> None:
    stdout = """
===== Run Summary - 2026-03-12 14:15:35 UTC =====
Runtimes:
    Total pages processed: 1940 from /datasets/nv-ingest/jp20
    Ingestion only time: 107.62s / 0:01:47.621
PPS:
    Ingestion only PPS: 18.03
Recall metrics:
  recall@1: 0.6261
  recall@5: 0.8261
  recall@10: 0.8870
Result: FAIL | return_code=98
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files is None
    assert metrics.pages == 1940
    assert metrics.ingest_secs == 107.62
    assert metrics.pages_per_sec_ingest == 18.03
    assert metrics.rows_processed is None
    assert metrics.rows_per_sec_ingest is None
    assert metrics.recall_metrics == {
        "recall@1": 0.6261,
        "recall@5": 0.8261,
        "recall@10": 0.8870,
    }


def test_parse_stream_text_extracts_current_run_summary_when_recall_is_skipped() -> None:
    stdout = """
===== Run Summary - 2026-03-12 14:45:11 UTC =====
Runtimes:
    Total pages processed: 496 from /datasets/nv-ingest/bo20
    Ingestion only time: 38.91s / 0:00:38.910
    Recall time: skipped
PPS:
    Ingestion only PPS: 12.75
    Recall QPS: skipped
Recall metrics: skipped
Result: PASS | return_code=0
"""
    metrics = parse_stream_text(stdout)
    assert metrics.files is None
    assert metrics.pages == 496
    assert metrics.ingest_secs == 38.91
    assert metrics.pages_per_sec_ingest == 12.75
    assert metrics.recall_metrics == {}


def test_stream_metrics_handles_non_recall_lines_after_recall_block() -> None:
    metrics = StreamMetrics()
    metrics.consume("Recall metrics:\n")
    metrics.consume("  recall@5: 0.9043\n")
    metrics.consume("Pages processed: 1933\n")
    metrics.consume("  recall@10: 0.9565\n")
    assert metrics.recall_metrics == {"recall@5": 0.9043}
