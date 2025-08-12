from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import nvingest_retrieval

import os
import time
import json
from datetime import datetime

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # Optional; stats logging will be skipped if unavailable


def _now_timestr() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _get_env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _default_collection_name() -> str:
    return f"dc20_{_now_timestr()}"


def _embed_info() -> tuple[str, int]:
    # Keep it simple and configurable; default matches compose's embedding NIM
    model_name = _get_env("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")
    # Known dims for common models; fall back to 2048
    if model_name.endswith("e5-v5"):
        return model_name, 1024
    return model_name, 2048


def _log_dict(d: dict) -> None:
    print(json.dumps(d))


def _milvus_chunks(collection_name: str) -> None:
    if MilvusClient is None:
        return
    try:
        client = MilvusClient(uri="http://localhost:19530")
        stats = client.get_collection_stats(collection_name)
        _log_dict({"metric": f"{collection_name}_chunks", "stats": stats})
        client.close()
    except Exception as e:
        _log_dict({"warning": f"milvus_stats_failed: {e}"})


def _segment_results(results):
    text_results = [[el for el in doc if el.get("document_type") == "text"] for doc in results]
    table_results = [
        [el for el in doc if el.get("metadata", {}).get("content_metadata", {}).get("subtype") == "table"]
        for doc in results
    ]
    chart_results = [
        [el for el in doc if el.get("metadata", {}).get("content_metadata", {}).get("subtype") == "chart"]
        for doc in results
    ]
    return text_results, table_results, chart_results


def main() -> int:
    data_dir = _get_env("DATASET_DIR", "/datasets/bo20")
    if not data_dir or not os.path.isdir(data_dir):
        print(f"DATASET_DIR does not exist: {data_dir}")
        return 2

    spill_dir = _get_env("SPILL_DIR", "/tmp/spill")
    os.makedirs(spill_dir, exist_ok=True)

    collection_name = _get_env("COLLECTION_NAME", _default_collection_name())
    hostname = "localhost"
    sparse = True
    gpu_search = False

    model_name, dense_dim = _embed_info()
    print(f"Embed model: {model_name}, dim: {dense_dim}")

    ingestion_start = time.time()
    ingestor = (
        Ingestor(message_client_hostname=hostname, message_client_port=7670)
        .files(data_dir)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
            text_depth="page",
            table_output_format="markdown",
            extract_infographics=True,
        )
        .embed(model_name=model_name)
        .vdb_upload(
            collection_name=collection_name,
            dense_dim=dense_dim,
            sparse=sparse,
            gpu_search=gpu_search,
            model_name=model_name,
        )
        .save_to_disk(output_directory=spill_dir)
    )

    results, failures = ingestor.ingest(show_progress=True, return_failures=True, save_to_disk=True)
    ingestion_time = time.time() - ingestion_start
    _log_dict({"metric": "result_count", "value": len(results)})
    _log_dict({"metric": "failure_count", "value": len(failures)})
    _log_dict({"metric": "ingestion_time_s", "value": ingestion_time})

    # Optional: log chunk stats and per-type breakdown
    _milvus_chunks(collection_name)
    text_results, table_results, chart_results = _segment_results(results)
    _log_dict({"metric": "text_chunks", "value": sum(len(x) for x in text_results)})
    _log_dict({"metric": "table_chunks", "value": sum(len(x) for x in table_results)})
    _log_dict({"metric": "chart_chunks", "value": sum(len(x) for x in chart_results)})

    # Retrieval sanity
    queries = [
        "What is the dog doing and where?",
        "How many dollars does a power drill cost?",
    ]
    querying_start = time.time()
    _ = nvingest_retrieval(
        queries,
        collection_name,
        hybrid=sparse,
        embedding_endpoint=f"http://{hostname}:8012/v1",
        embedding_model_name=model_name,
        model_name=model_name,
        top_k=5,
        gpu_search=gpu_search,
    )
    _log_dict({"metric": "retrieval_time_s", "value": time.time() - querying_start})

    # Summarize
    summary = {
        "dataset_dir": data_dir,
        "collection_name": collection_name,
        "model_name": model_name,
        "dense_dim": dense_dim,
        "ingestion_time_s": ingestion_time,
        "result_count": len(results),
        "failure_count": len(failures),
    }
    print("dc20_e2e summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


