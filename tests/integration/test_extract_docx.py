import pytest
import time
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from tests.integration.utilities_for_test import canonicalize_markdown_table
import threading
import queue


def _run_with_timeout(callable_fn, timeout_s: float):
    """Run callable_fn() with a hard timeout; return its result or raise TimeoutError."""
    q: queue.Queue = queue.Queue(maxsize=1)

    def _target():
        try:
            q.put((True, callable_fn()))
        except Exception as e:  # surface exceptions to main thread
            q.put((False, e))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_s} seconds")
    ok, payload = q.get_nowait()
    if ok:
        return payload
    raise payload


def _debug_summarize_results(results) -> str:
    """Return a concise string summary of results for debugging."""
    try:
        doc_count = len(results) if isinstance(results, list) else 0
        if doc_count == 0:
            return "[DEBUG] No results returned"
        items = results[0] if isinstance(results[0], list) else []
        total_items = len(items)
        structured = [
            x
            for x in items
            if isinstance(x, dict) and x.get("metadata", {}).get("content_metadata", {}).get("type") == "structured"
        ]
        counts = {}
        for x in structured:
            st = x.get("metadata", {}).get("content_metadata", {}).get("subtype")
            counts[st] = counts.get(st, 0) + 1
        lines = [
            f"[DEBUG] Results summary: docs={doc_count}, items_in_doc0={total_items}, structured={len(structured)}",
            f"[DEBUG] Structured breakdown: {counts}",
        ]
        # include a small sample of first 2 structured entries' types
        sample = structured[:2]
        sample_types = [x.get("metadata", {}).get("content_metadata", {}).get("subtype") for x in sample]
        lines.append(f"[DEBUG] Sample structured subtypes (first2): {sample_types}")
        return "\n".join(lines)
    except Exception as e:
        return f"[DEBUG] Failed to summarize results: {e}"


@pytest.mark.integration
def test_docx_extract_only(
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_first_chart_xaxis,
    multimodal_first_chart_yaxis,
    multimodal_second_chart_xaxis,
    multimodal_second_chart_yaxis,
):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
        worker_pool_size=2,
    )

    results = None
    try:
        start_time = time.time()
        ingestor = (
            Ingestor(client=client)
            .files("./data/multimodal_test.docx")
            .extract(
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                table_output_format="markdown",
                extract_infographics=True,
                text_depth="page",
            )
        )

        # Align with example behavior but enforce finite retries/timeouts to avoid infinite hangs
        def _do_ingest():
            return ingestor.ingest(
                show_progress=False,
                return_failures=True,
                timeout=60,  # per-batch wait for futures
                max_job_retries=3,  # bound fetch retries
                verbose=False,
            )

        # Bound total ingest time to avoid hanging CI
        try:
            results_and_failures = _run_with_timeout(_do_ingest, timeout_s=180.0)
        except Exception as e:
            print(f"[DEBUG] Ingest raised exception: {e!r}")
            raise
        # ingest(show_progress=False, return_failures=True) returns (results, failures)
        if isinstance(results_and_failures, tuple) and len(results_and_failures) == 2:
            results, failures = results_and_failures
        else:
            results, failures = results_and_failures, []
        if failures:
            print(_debug_summarize_results(results))
            pytest.fail(f"Ingestion reported failures: {failures}")
        results = results
    finally:
        # Ensure the client's worker threads do not keep pytest alive on errors/timeouts
        try:
            client._worker_pool.shutdown(wait=False)  # type: ignore[attr-defined]
        except Exception:
            pass
    duration = time.time() - start_time
    print(f"[DEBUG] Ingest completed in {duration:.1f}s")
    print(_debug_summarize_results(results))

    assert len(results) == 1, f"Expected 1 document, got {len(results)}"

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) >= 3, f"Expected >=3 structured items, got {len(structured)}"

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) >= 1, f"Expected >=1 tables, got {len(tables)}"
    assert len(charts) >= 2, f"Expected >=2 charts, got {len(charts)}"
    assert len(infographics) == 0, f"Expected 0 infographics, got {len(infographics)}"

    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert any(
        canonicalize_markdown_table(x) == multimodal_first_table_markdown for x in table_contents
    ), "Expected canonical table not found in table contents"

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    assert multimodal_first_chart_xaxis in chart_contents, "First chart x-axis label not found"
    assert multimodal_first_chart_yaxis in chart_contents, "First chart y-axis label not found"
    assert multimodal_second_chart_xaxis in chart_contents, "Second chart x-axis label not found"
    assert (
        multimodal_second_chart_yaxis.replace("1 - ", "") in chart_contents
    ), "Second chart y-axis label not found (after normalization)"
