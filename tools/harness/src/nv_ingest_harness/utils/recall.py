"""
Recall evaluation utilities for nv-ingest test cases.

Provides dataset-specific evaluators and helper functions for recall testing.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Callable

from nv_ingest_client.util.milvus import nvingest_retrieval

from nv_ingest_harness.utils.cases import get_repo_root


def _get_retrieval_func(
    vdb_backend: str,
    table_path: Optional[str] = None,
    table_name: Optional[str] = None,
):
    """
    Get the retrieval function for the specified VDB backend.

    For LanceDB, returns a partial of lancedb_retrieval with table_path pre-filled.
    For Milvus, returns nvingest_retrieval directly.

    Args:
        vdb_backend: "milvus" or "lancedb"
        table_path: Path to LanceDB database directory (required for lancedb)
        table_name: Table/collection name (optional, can be passed at call time)

    Returns:
        Retrieval function that accepts (queries, **kwargs) and returns results.
    """
    if vdb_backend == "lancedb":
        if not table_path:
            raise ValueError("table_path required for lancedb backend")
        from nv_ingest_client.util.vdb.lancedb import lancedb_retrieval

        kwargs = {"table_path": table_path}
        if table_name:
            kwargs["table_name"] = table_name
        return partial(lancedb_retrieval, **kwargs)

    return nvingest_retrieval


def get_recall_scores(
    query_df: pd.DataFrame,
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
    batch_size: int = 100,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Calculate recall@k scores for queries against a VDB collection.

    Matches the notebook evaluation pattern: extracts pdf_page identifiers from retrieved
    results and checks if the expected pdf_page appears in the top-k results.

    Args:
        query_df: DataFrame with required columns 'query' and 'pdf_page'.
                  pdf_page format: '{pdf_id}_{page_number}' (1-indexed page numbers).
        collection_name: Collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True (Milvus only).
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking (Milvus only).
        ground_truth_dir: Unused parameter (kept for API compatibility).
        batch_size: Number of queries to process per batch (prevents gRPC size limit errors).
        vdb_backend: VDB backend to use ("milvus" or "lancedb"). Default is "milvus".
        table_path: Path to LanceDB database directory (required if vdb_backend="lancedb").

    Returns:
        Dictionary mapping k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
    """
    hits = defaultdict(list)
    queries = query_df["query"].to_list()
    pdf_pages = query_df["pdf_page"].to_list()
    num_queries = len(queries)
    num_batches = (num_queries + batch_size - 1) // batch_size

    # Create retrieval function once, outside the batch loop
    if vdb_backend == "lancedb":
        retrieval_func = _get_retrieval_func("lancedb", table_path, table_name=collection_name)
    else:
        retrieval_func = None  # Use nvingest_retrieval directly

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_queries)
        batch_queries = queries[start_idx:end_idx]
        batch_pdf_pages = pdf_pages[start_idx:end_idx]

        if num_batches > 1:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({len(batch_queries)} queries)")

        if vdb_backend == "lancedb":
            batch_answers = retrieval_func(
                batch_queries,
                embedding_endpoint=f"http://{hostname}:8012/v1",
                model_name=model_name,
                top_k=top_k,
                nv_ranker=nv_ranker,
                nv_ranker_endpoint=nv_ranker_endpoint,
                nv_ranker_model_name=nv_ranker_model_name,
            )
        else:
            batch_answers = nvingest_retrieval(
                batch_queries,
                collection_name,
                hybrid=sparse,
                embedding_endpoint=f"http://{hostname}:8012/v1",
                model_name=model_name,
                top_k=top_k,
                gpu_search=gpu_search,
                nv_ranker=nv_ranker,
            )

        for expected_pdf_page, retrieved_answers in zip(batch_pdf_pages, batch_answers):
            retrieved_pdf_pages = []
            for result in retrieved_answers:
                source_id = result.get("entity", {}).get("source", {}).get("source_id", "")
                content_metadata = result.get("entity", {}).get("content_metadata", {})
                page_number = content_metadata.get("page_number", "")
                pdf_name = os.path.basename(source_id).split(".")[0]
                retrieved_pdf_pages.append(f"{pdf_name}_{page_number}")

            for k in [1, 3, 5, 10]:
                if k <= top_k:
                    hits[k].append(expected_pdf_page in retrieved_pdf_pages[:k])

    recall_scores = {k: np.mean(hits[k]) for k in hits if len(hits[k]) > 0}

    return recall_scores


def get_recall_scores_pdf_only(
    query_df: pd.DataFrame,
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = False,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
    batch_size: int = 100,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Calculate recall@k scores for queries against a VDB collection using PDF-only matching.

    Matches only PDF filenames (no page numbers), used for datasets like finance_bench where
    ground truth is at document level rather than page level.

    Args:
        query_df: DataFrame with required columns 'query' and 'expected_pdf'.
                  expected_pdf format: PDF filename without extension (e.g., '3M_2018_10K').
        collection_name: Collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True (Milvus only).
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking (Milvus only).
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        ground_truth_dir: Unused parameter (kept for API compatibility).
        batch_size: Number of queries to process per batch (prevents gRPC size limit errors).
        vdb_backend: VDB backend to use ("milvus" or "lancedb"). Default is "milvus".
        table_path: Path to LanceDB database directory (required if vdb_backend="lancedb").

    Returns:
        Dictionary mapping k values (1, 5, 10) to recall scores (float 0.0-1.0).
    """
    hits = defaultdict(list)

    reranker_kwargs = {}
    if nv_ranker:
        if nv_ranker_endpoint:
            reranker_kwargs["nv_ranker_endpoint"] = nv_ranker_endpoint
        if nv_ranker_model_name:
            reranker_kwargs["nv_ranker_model_name"] = nv_ranker_model_name

    queries = query_df["query"].to_list()
    expected_pdfs = query_df["expected_pdf"].to_list()

    # Process queries in batches to avoid gRPC message size limits
    num_queries = len(queries)
    num_batches = (num_queries + batch_size - 1) // batch_size

    # Create retrieval function once, outside the batch loop
    if vdb_backend == "lancedb":
        retrieval_func = _get_retrieval_func("lancedb", table_path, table_name=collection_name)
    else:
        retrieval_func = None  # Use nvingest_retrieval directly

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_queries)
        batch_queries = queries[start_idx:end_idx]
        batch_expected_pdfs = expected_pdfs[start_idx:end_idx]

        if num_batches > 1:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({len(batch_queries)} queries)")

        if vdb_backend == "lancedb":
            batch_answers = retrieval_func(
                batch_queries,
                embedding_endpoint=f"http://{hostname}:8012/v1",
                model_name=model_name,
                top_k=top_k,
                nv_ranker=nv_ranker,
                nv_ranker_endpoint=nv_ranker_endpoint,
                nv_ranker_model_name=nv_ranker_model_name,
            )
        else:
            batch_answers = nvingest_retrieval(
                batch_queries,
                collection_name,
                hybrid=sparse,
                embedding_endpoint=f"http://{hostname}:8012/v1",
                model_name=model_name,
                top_k=top_k,
                gpu_search=gpu_search,
                nv_ranker=nv_ranker,
                **reranker_kwargs,
            )

        for expected_pdf, retrieved_answers in zip(batch_expected_pdfs, batch_answers):
            # Extract PDF names only (no page numbers)
            retrieved_pdfs = [
                os.path.basename(result.get("entity", {}).get("source", {}).get("source_id", "")).split(".")[0]
                for result in retrieved_answers
            ]

            # Finance_bench uses k values [1, 5, 10]
            for k in [1, 5, 10]:
                if k <= top_k:
                    hits[k].append(expected_pdf in retrieved_pdfs[:k])

    recall_scores = {k: np.mean(hits[k]) for k in hits if len(hits[k]) > 0}

    return recall_scores


def evaluate_recall_orchestrator(
    loader_func: Callable[[Optional[str]], pd.DataFrame],
    scorer_func: Callable,
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = False,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
    **scorer_kwargs,
) -> Dict[int, float]:
    """
    Generic orchestrator for recall evaluation.

    Args:
        loader_func: Function that loads ground truth DataFrame.
        scorer_func: Function that calculates recall scores.
        collection_name: Collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid retrieval (Milvus only).
        model_name: Embedding model name.
        top_k: Maximum results to retrieve.
        gpu_search: Use GPU for search (Milvus only).
        nv_ranker: Enable reranker.
        ground_truth_dir: Directory containing ground truth files.
        vdb_backend: VDB backend ("milvus" or "lancedb").
        table_path: Path to LanceDB database.
        **scorer_kwargs: Additional kwargs for scorer_func.

    Returns:
        Dictionary mapping k values to recall scores.
    """
    query_df = loader_func(ground_truth_dir)

    scores = scorer_func(
        query_df,
        collection_name,
        hostname=hostname,
        sparse=sparse,
        vdb_backend=vdb_backend,
        table_path=table_path,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        **scorer_kwargs,
    )

    # 3. Return scores
    return scores


def bo767_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load bo767 ground truth queries from consolidated CSV file.

    All queries are multimodal (no modality filtering).

    Args:
        ground_truth_dir: Directory containing bo767_query_gt.csv.
                         Defaults to repo data/ directory if None.

    Returns:
        DataFrame with columns: query, pdf, page, modality, pdf_page.
        pdf_page format: '{pdf_id}_{page_number}' (1-indexed page numbers).
    """
    if ground_truth_dir is None:
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    csv_path = os.path.join(ground_truth_dir, "bo767_query_gt.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"bo767_query_gt.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "pdf" in df.columns:
        df["pdf"] = df["pdf"].astype(str).apply(lambda x: x.replace(".pdf", ""))

    return df.reset_index(drop=True)


def bo767_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Evaluate recall@k for bo767 dataset (multimodal-only).

    Loads ground truth queries from bo767_query_gt.csv and evaluates recall
    against the specified VDB collection.

    Args:
        collection_name: VDB collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True (Milvus only).
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing bo767_query_gt.csv (optional).
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        vdb_backend: VDB backend to use ("milvus" or "lancedb"). Default is "milvus".
        table_path: Path to LanceDB database directory (required if vdb_backend="lancedb").

    Returns:
        Dictionary mapping k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
        Only includes k values where k <= top_k.
    """
    return evaluate_recall_orchestrator(
        loader_func=bo767_load_ground_truth,
        scorer_func=get_recall_scores,
        collection_name=collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def finance_bench_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load ground truth queries for finance_bench dataset (multimodal-only).

    Args:
        ground_truth_dir: Directory containing ground truth JSON file.
                         If None, uses repo data/ directory.

    Returns:
        DataFrame with columns: 'query' (from 'question'), 'expected_pdf' (from contexts[0]['filename'])
    """
    if ground_truth_dir is None:
        # Default to repo data/ directory
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    # Load finance_bench JSON - expected format: list of dicts with 'question' and 'contexts'
    json_path = os.path.join(ground_truth_dir, "financebench_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"finance_bench ground truth file not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract questions and expected PDFs
    queries = []
    expected_pdfs = []
    for item in data:
        if "question" not in item:
            continue
        if "contexts" not in item or len(item["contexts"]) == 0:
            continue
        if "filename" not in item["contexts"][0]:
            continue

        queries.append(item["question"])
        # Extract filename without extension
        filename = item["contexts"][0]["filename"]
        # Remove .pdf extension if present
        expected_pdf = filename.replace(".pdf", "") if isinstance(filename, str) else filename
        expected_pdfs.append(expected_pdf)

    if len(queries) == 0:
        raise ValueError(f"No valid queries found in {json_path}")

    return pd.DataFrame({"query": queries, "expected_pdf": expected_pdfs})


def finance_bench_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = False,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Evaluate recall@k for finance_bench dataset (multimodal-only).

    Loads ground truth queries from financebench_train.json and evaluates recall
    using PDF-only matching (document level, not page level).

    Args:
        collection_name: VDB collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True (Milvus only).
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing financebench_train.json (optional).
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        vdb_backend: VDB backend to use ("milvus" or "lancedb"). Default is "milvus".
        table_path: Path to LanceDB database directory (required if vdb_backend="lancedb").

    Returns:
        Dictionary mapping k values (1, 5, 10) to recall scores (float 0.0-1.0).
        Only includes k values where k <= top_k.
    """
    return evaluate_recall_orchestrator(
        loader_func=finance_bench_load_ground_truth,
        scorer_func=get_recall_scores_pdf_only,
        collection_name=collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def earnings_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load ground truth queries for earnings dataset (multimodal-only).

    The earnings CSV uses 0-indexed page numbers, which are converted to 1-indexed
    in the pdf_page column to match Milvus storage format.

    Args:
        ground_truth_dir: Directory containing ground truth CSV file.
                         If None, uses repo data/ directory.

    Returns:
        DataFrame with columns: query, pdf, page, pdf_page
        pdf_page format: '{pdf_id}_{1_indexed_page_number}' (pages converted from 0-indexed CSV)
    """
    if ground_truth_dir is None:
        # Default to repo data/ directory
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    # Load earnings CSV - expected format: dir, pdf, page, query, answer, modality
    csv_path = os.path.join(ground_truth_dir, "earnings_consulting_multimodal.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"earnings ground truth file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["query", "pdf", "page"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"earnings CSV missing required columns: {missing_cols}")

    # Clean PDF names (remove .pdf extension if present)
    if "pdf" in df.columns:
        df["pdf"] = df["pdf"].apply(lambda x: str(x).replace(".pdf", "") if pd.notna(x) else x)

    # Create pdf_page column: convert 0-indexed CSV pages to 1-indexed for Milvus matching
    # Earnings CSV uses 0-indexed pages, but Milvus stores 1-indexed
    # Format: '{pdf_id}_{1_indexed_page_number}'
    df["pdf_page"] = df.apply(lambda x: f"{x.pdf}_{x.page + 1}", axis=1)

    # Keep only necessary columns for recall evaluation
    return df[["query", "pdf", "page", "pdf_page"]].copy().reset_index(drop=True)


def earnings_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Evaluate recall@k for earnings dataset (multimodal-only).

    Loads ground truth queries from earnings_consulting_multimodal.csv and evaluates recall
    using PDF+page matching.

    Args:
        collection_name: VDB collection/table name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True (Milvus only).
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing earnings_consulting_multimodal.csv (optional).
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        vdb_backend: VDB backend to use ("milvus" or "lancedb"). Default is "milvus".
        table_path: Path to LanceDB database directory (required if vdb_backend="lancedb").

    Returns:
        Dictionary mapping k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
        Only includes k values where k <= top_k.
    """
    return evaluate_recall_orchestrator(
        loader_func=earnings_load_ground_truth,
        scorer_func=get_recall_scores,
        collection_name=collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def audio_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load audio retrieval ground truth from video_retrieval_eval_gt.csv.

    Filters for audio-only questions (answer_modality == 'Audio only').

    Args:
        ground_truth_dir: Directory containing video_retrieval_eval_gt.csv.
                         Defaults to repo data/ directory if None.

    Returns:
        DataFrame with columns: query, expected_video, start_time, end_time
        (filtered for audio-only questions)
    """
    if ground_truth_dir is None:
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    csv_path = os.path.join(ground_truth_dir, "video_retrieval_eval_gt.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"video_retrieval_eval_gt.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter for audio-only questions
    df = df.query("answer_modality == 'Audio only'").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No 'Audio only' questions found in video_retrieval_eval_gt.csv")

    # Rename columns to match expected format
    df = df.rename(columns={"name": "expected_video", "question": "query"})

    # Keep relevant columns
    return df[["query", "expected_video", "start_time", "end_time"]].copy().reset_index(drop=True)


def get_recall_scores_audio(
    query_df: pd.DataFrame,
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = False,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
    tolerance: float = 2.0,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Calculate recall@k scores for audio dataset using segment-level matching.

    Uses chunk midpoint matching with time tolerance (aligned with audio_recall.py).
    A segment-level hit requires both audio file match AND midpoint within ground truth time range.

    Args:
        query_df: DataFrame with columns 'query', 'expected_video', 'start_time', 'end_time'.
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        ground_truth_dir: Unused parameter (kept for API compatibility).
        tolerance: Time tolerance in seconds for segment matching (default 2.0).

    Returns:
        Dictionary mapping k values (1, 5, 10) to segment-level recall scores.
    """
    hits = defaultdict(list)

    # Prepare reranker kwargs - match original audio_recall.py defaults
    reranker_kwargs = {}
    if nv_ranker:
        # Use provided endpoint/model or fall back to defaults from original script
        reranker_kwargs["nv_ranker_endpoint"] = nv_ranker_endpoint or f"http://{hostname}:8020/v1/ranking"
        reranker_kwargs["nv_ranker_model_name"] = nv_ranker_model_name or "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        # Fetch 50 results before reranking to top_k (matches original audio_recall.py)
        reranker_kwargs["nv_ranker_top_k"] = 50

    all_answers = nvingest_retrieval(
        query_df["query"].to_list(),
        collection_name,
        hybrid=sparse,
        embedding_endpoint=f"http://{hostname}:8012/v1",
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        **reranker_kwargs,
    )

    for i in range(len(query_df)):
        expected_audio = query_df["expected_video"].iloc[i]
        gt_start = query_df["start_time"].iloc[i]
        gt_end = query_df["end_time"].iloc[i]
        retrieved_answers = all_answers[i]

        # Extract audio file names and time info from retrieved chunks
        retrieved_audios = []
        retrieved_midpoints = []
        for result in retrieved_answers:
            source_id = result.get("entity", {}).get("source", {}).get("source_id", "")
            content_metadata = result.get("entity", {}).get("content_metadata", {})

            # Extract audio name (remove path and extension)
            audio_name = os.path.basename(source_id).split(".")[0]
            retrieved_audios.append(audio_name)

            # Extract time range and calculate midpoint
            # Milvus stores time in milliseconds, convert to seconds
            start_time_ms = float(content_metadata.get("start_time", 0))
            end_time_ms = float(content_metadata.get("end_time", 0))
            start_time_sec = start_time_ms / 1000
            end_time_sec = end_time_ms / 1000
            midpoint = (start_time_sec + end_time_sec) / 2
            retrieved_midpoints.append(midpoint)

        # Segment-level matching: audio + time overlap with tolerance
        for k in [1, 5, 10]:
            if k <= top_k:
                in_topk = False
                for j in range(min(k, len(retrieved_audios))):
                    audio_matches = retrieved_audios[j] == expected_audio
                    time_matches = (gt_start - tolerance) <= retrieved_midpoints[j] <= (gt_end + tolerance)
                    if audio_matches and time_matches:
                        in_topk = True
                        break
                hits[k].append(in_topk)

    recall_scores = {k: np.mean(hits[k]) for k in hits if len(hits[k]) > 0}
    return recall_scores


def audio_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = False,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Evaluate recall@k for audio dataset.

    Loads ground truth from video_retrieval_eval_gt.csv filtered for 'Audio only' modality
    and evaluates recall using segment-level matching (audio file name + time overlap with 2s tolerance).

    Args:
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing video_retrieval_eval_gt.csv (optional).
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.

    Returns:
        Dictionary mapping k values (1, 5, 10) to segment-level recall scores.
    """
    # Load ground truth (filtered for audio-only)
    query_df = audio_load_ground_truth(ground_truth_dir)

    print(f"Audio recall: {len(query_df)} audio-only questions loaded")

    # Calculate recall scores
    return get_recall_scores_audio(
        query_df,
        collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def video_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load video retrieval ground truth from CSV.

    Args:
        ground_truth_dir: Directory containing video_retrieval_eval_gt.csv.
                         Defaults to repo data/ directory if None.

    Returns:
        DataFrame with columns: query, expected_video, start_time, end_time
    """
    if ground_truth_dir is None:
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    csv_path = os.path.join(ground_truth_dir, "video_retrieval_eval_gt.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"video_retrieval_eval_gt.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename columns to match expected format
    df = df.rename(columns={"name": "expected_video", "question": "query"})

    # Keep relevant columns
    return df[["query", "expected_video", "start_time", "end_time"]].copy().reset_index(drop=True)


def get_recall_scores_video(
    query_df: pd.DataFrame,
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    ground_truth_dir: Optional[str] = None,
    tolerance: float = 2.0,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Calculate recall@k scores for video dataset using segment-level matching.

    Uses chunk midpoint matching with time tolerance (aligned with historical evaluation.py).
    A segment-level hit requires both video match AND midpoint within ground truth time range.
    Video-level recall (informational) is printed but segment-level is returned as primary.

    Automatically filters ground truth to only include queries for videos in the collection,
    enabling meaningful recall evaluation with partial datasets.

    Args:
        query_df: DataFrame with columns 'query', 'expected_video', 'start_time', 'end_time'.
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Unused parameter (kept for API compatibility).
        tolerance: Time tolerance in seconds for segment matching (default 2.0).

    Returns:
        Dictionary mapping k values (1, 5, 10) to segment-level recall scores.
        Video-level scores are printed separately as informational.
    """
    # First, discover which videos are actually in the collection
    # by running a simple query to get all unique source_ids
    from pymilvus import Collection, connections

    milvus_uri = f"http://{hostname}:19530"
    connections.connect(alias="default", uri=milvus_uri)
    collection = Collection(collection_name)

    # Query to get unique video names in collection
    results = collection.query(
        expr="",
        output_fields=["source"],
        limit=16384,  # Get all chunks
    )
    videos_in_collection = set()
    for r in results:
        source_id = r.get("source", {}).get("source_id", "")
        video_name = os.path.basename(source_id).split(".")[0]
        if video_name:
            videos_in_collection.add(video_name)

    print(f"\nVideos in collection: {len(videos_in_collection)}")

    # Filter query_df to only include queries for videos in the collection
    original_count = len(query_df)
    query_df_filtered = query_df[query_df["expected_video"].isin(videos_in_collection)].reset_index(drop=True)
    filtered_count = len(query_df_filtered)

    print(f"Ground truth queries: {original_count} total, {filtered_count} matching videos in collection")

    if filtered_count == 0:
        print("WARNING: No ground truth queries match videos in collection!")
        print(f"  Sample GT videos: {list(query_df['expected_video'].unique()[:5])}")
        print(f"  Sample collection videos: {list(videos_in_collection)[:5]}")
        return {1: 0.0, 5: 0.0, 10: 0.0}

    # Use filtered dataframe for evaluation
    query_df = query_df_filtered

    segment_hits = defaultdict(list)
    video_hits = defaultdict(list)

    all_answers = nvingest_retrieval(
        query_df["query"].to_list(),
        collection_name,
        hybrid=sparse,
        embedding_endpoint=f"http://{hostname}:8012/v1",
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
    )

    # Debug: print first few queries to verify time values
    debug_limit = 3

    for i in range(len(query_df)):
        expected_video = query_df["expected_video"].iloc[i]
        gt_start = query_df["start_time"].iloc[i]
        gt_end = query_df["end_time"].iloc[i]
        retrieved_answers = all_answers[i]

        # Extract video names and time info from retrieved chunks
        retrieved_videos = []
        retrieved_midpoints = []
        for result in retrieved_answers:
            source_id = result.get("entity", {}).get("source", {}).get("source_id", "")
            content_metadata = result.get("entity", {}).get("content_metadata", {})

            # Extract video name (remove path and extension)
            video_name = os.path.basename(source_id).split(".")[0]
            retrieved_videos.append(video_name)

            # Extract time range and calculate midpoint
            # Milvus stores time in milliseconds, convert to seconds
            start_time_ms = float(content_metadata.get("start_time", 0))
            end_time_ms = float(content_metadata.get("end_time", 0))
            start_time_sec = start_time_ms / 1000
            end_time_sec = end_time_ms / 1000
            midpoint = (start_time_sec + end_time_sec) / 2
            retrieved_midpoints.append(midpoint)

        # Debug: print sample time values to verify conversion
        if i < debug_limit and len(retrieved_videos) > 0:
            print(f"\n[DEBUG] Query {i}: expected={expected_video}, gt_time=[{gt_start:.2f}, {gt_end:.2f}]s")
            for j in range(min(3, len(retrieved_videos))):
                print(f"  Result {j}: video={retrieved_videos[j]}, midpoint={retrieved_midpoints[j]:.2f}s")

        # Segment-level matching: video + time overlap with tolerance
        for k in [1, 5, 10]:
            if k <= top_k:
                in_topk = False
                for j in range(min(k, len(retrieved_videos))):
                    video_matches = retrieved_videos[j] == expected_video
                    time_matches = (gt_start - tolerance) <= retrieved_midpoints[j] <= (gt_end + tolerance)
                    if video_matches and time_matches:
                        in_topk = True
                        break
                segment_hits[k].append(in_topk)

        # Video-level matching: deduplicate by video, check if expected video in top-k
        seen_videos = []
        for video in retrieved_videos:
            if video not in seen_videos:
                seen_videos.append(video)

        for k in [1, 5, 10]:
            if k <= top_k:
                video_hits[k].append(expected_video in seen_videos[:k])

    segment_scores = {k: np.mean(segment_hits[k]) for k in segment_hits if len(segment_hits[k]) > 0}
    video_scores = {k: np.mean(video_hits[k]) for k in video_hits if len(video_hits[k]) > 0}

    # Print video-level recall as informational metric
    print("\nVideo-level Recall (informational):")
    for k in sorted(video_scores.keys()):
        print(f"  - Video Recall @{k}: {video_scores[k]:.3f}")

    # Return segment-level as primary metric (matches recall case expectations)
    return segment_scores


def video_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """
    Evaluate recall@k for video dataset.

    Loads ground truth from video_retrieval_eval_gt.csv and evaluates recall using:
    - Segment-level matching (primary, returned): video name + time overlap with 2s tolerance
    - Video-level matching (informational, printed): video name only (deduplicated)

    Args:
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing video_retrieval_eval_gt.csv (optional).

    Returns:
        Dictionary mapping k values (1, 5, 10) to segment-level recall scores.
        Video-level scores are printed as informational during evaluation.
    """
    # Load ground truth
    query_df = video_load_ground_truth(ground_truth_dir)

    # Calculate recall scores
    return get_recall_scores_video(
        query_df,
        collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        ground_truth_dir=ground_truth_dir,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def bo10k_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """Load bo10k ground truth from digital_corpora_10k_annotations.csv."""
    if ground_truth_dir is None:
        ground_truth_dir = os.path.join(get_repo_root(), "data")

    csv_path = os.path.join(ground_truth_dir, "digital_corpora_10k_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"digital_corpora_10k_annotations.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["query", "pdf", "page"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"bo10k CSV missing required columns: {missing_cols}")

    df["pdf"] = df["pdf"].astype(str).apply(lambda x: x.replace(".pdf", ""))
    df["pdf_page"] = df.apply(lambda x: f"{x.pdf}_{x.page + 1}", axis=1)

    return df.reset_index(drop=True)


def bo10k_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
    nv_ranker_endpoint: Optional[str] = None,
    nv_ranker_model_name: Optional[str] = None,
    vdb_backend: str = "milvus",
    table_path: Optional[str] = None,
) -> Dict[int, float]:
    """Evaluate recall@k for bo10k dataset."""
    return evaluate_recall_orchestrator(
        loader_func=bo10k_load_ground_truth,
        scorer_func=get_recall_scores,
        collection_name=collection_name,
        hostname=hostname,
        sparse=sparse,
        model_name=model_name,
        top_k=top_k,
        gpu_search=gpu_search,
        nv_ranker=nv_ranker,
        ground_truth_dir=ground_truth_dir,
        nv_ranker_endpoint=nv_ranker_endpoint,
        nv_ranker_model_name=nv_ranker_model_name,
        vdb_backend=vdb_backend,
        table_path=table_path,
    )


def get_recall_collection_name(test_name: str) -> str:
    """
    Generate collection name for recall evaluation.

    For normal datasets, uses the pattern {test_name}_multimodal.
    If the input is a single file (contains '.' or ends with '.pdf' or similar),
    it converts the period, which is not allowed in Milvus collection names, to an underscore.

    Args:
        test_name: Test identifier (e.g., 'bo767', 'finance_bench', 'somefilename.pdf')

    Returns:
        Collection name string (e.g., 'bo767_multimodal', 'somefilename_pdf_multimodal')
    """
    sanitized_name = test_name.replace(".", "_")
    return f"{sanitized_name}_multimodal"


def get_dataset_evaluator(dataset_name: str) -> Optional[Callable]:
    """
    Get the recall evaluator function for a given dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'bo767', 'finance_bench')

    Returns:
        Evaluator function or None if not found
    """
    evaluators = {
        "bo767": bo767_recall,
        "finance_bench": finance_bench_recall,
        "earnings": earnings_recall,
        "audio": audio_recall,
        "video": video_recall,
        "bo10k": bo10k_recall,
    }

    return evaluators.get(dataset_name.lower())
