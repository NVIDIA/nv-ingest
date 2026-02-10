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
    hybrid: bool = False,
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
        if hybrid:
            from nv_ingest_client.util.vdb.lancedb import lancedb_hybrid_retrieval as lancedb_retrieval
        else:
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
    hybrid: bool = False,
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
        retrieval_func = _get_retrieval_func(
            "lancedb",
            table_path,
            table_name=collection_name,
            hybrid=hybrid,
        )
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
    hybrid: bool = False,
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
        retrieval_func = _get_retrieval_func(
            "lancedb",
            table_path,
            table_name=collection_name,
            hybrid=hybrid,
        )
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
    hybrid: bool = False,
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
        hybrid=hybrid,
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
    hybrid: bool = False,
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
        hybrid=hybrid,
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
    hybrid: bool = False,
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
        hybrid=hybrid,
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
    hybrid: bool = False,
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
        hybrid=hybrid,
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


def audio_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
) -> Dict[int, float]:
    """
    Audio dataset recall evaluator (stub, multimodal-only).

    TODO: Implement audio-specific ground truth loading and evaluation.
    """
    raise NotImplementedError("audio_recall not yet implemented")


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


def jp20_load_ground_truth(ground_truth_dir: Optional[str] = None) -> pd.DataFrame:
    """Load bo10k ground truth filtered to jp20 documents."""
    df = bo10k_load_ground_truth(ground_truth_dir=ground_truth_dir)

    dataset_dir = os.environ.get("JP20_DATASET_DIR") or os.path.join(get_repo_root(), "data", "jp20")
    jp20_pdfs = {os.path.splitext(name)[0] for name in os.listdir(dataset_dir) if name.lower().endswith(".pdf")}
    filtered = df[df["pdf"].isin(jp20_pdfs)].reset_index(drop=True)

    return filtered


def bo10k_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    hybrid: bool = False,
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
        hybrid=hybrid,
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


def jp20_recall(
    collection_name: str,
    hostname: str = "localhost",
    sparse: bool = True,
    hybrid: bool = False,
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
    """Evaluate recall@k for jp20 dataset (bo10k subset)."""
    return evaluate_recall_orchestrator(
        loader_func=jp20_load_ground_truth,
        scorer_func=get_recall_scores,
        collection_name=collection_name,
        hostname=hostname,
        sparse=sparse,
        hybrid=hybrid,
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
        "bo10k": bo10k_recall,
        "jp20": jp20_recall,
    }

    return evaluators.get(dataset_name.lower())
