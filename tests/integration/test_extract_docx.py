import pytest
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
        message_client_hostname="127.0.0.1",
        worker_pool_size=2,
    )

    try:
        ingestor = (
            Ingestor(client=client)
            .files("./data/multimodal_test.docx")
            .extract(
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                paddle_output_format="markdown",
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

        results_and_failures = _run_with_timeout(_do_ingest, timeout_s=180.0)
        # ingest(show_progress=False, return_failures=True) returns (results, failures)
        if isinstance(results_and_failures, tuple) and len(results_and_failures) == 2:
            results, failures = results_and_failures
        else:
            results, failures = results_and_failures, []
        if failures:
            pytest.fail(f"Ingestion reported failures: {failures}")
        results = results
    finally:
        # Ensure the client's worker threads do not keep pytest alive on errors/timeouts
        try:
            client._worker_pool.shutdown(wait=False)  # type: ignore[attr-defined]
        except Exception:
            pass
    assert len(results) == 1

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) >= 3

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) >= 1
    assert len(charts) >= 2
    assert len(infographics) == 0

    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert any(canonicalize_markdown_table(x) == multimodal_first_table_markdown for x in table_contents)

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    assert multimodal_first_chart_xaxis in chart_contents
    assert multimodal_first_chart_yaxis in chart_contents
    assert multimodal_second_chart_xaxis in chart_contents
    assert multimodal_second_chart_yaxis.replace("1 - ", "") in chart_contents
