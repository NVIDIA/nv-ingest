"""
Recall evaluation utilities for nv-ingest test cases.

Provides dataset-specific evaluators and helper functions for recall testing.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Callable

from nv_ingest_client.util.milvus import nvingest_retrieval


def get_recall_scores(
    query_df: pd.DataFrame,
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
    Calculate recall@k scores for queries against a Milvus collection.

    Matches the notebook evaluation pattern: extracts pdf_page identifiers from retrieved
    results and checks if the expected pdf_page appears in the top-k results.

    Args:
        query_df: DataFrame with required columns 'query' and 'pdf_page'.
                  pdf_page format: '{pdf_id}_{page_number}' (1-indexed page numbers).
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Unused parameter (kept for API compatibility).

    Returns:
        Dictionary mapping k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
        Only includes k values where k <= top_k.
    """
    hits = defaultdict(list)

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

    for i in range(len(query_df)):
        expected_pdf_page = query_df["pdf_page"].iloc[i]
        retrieved_answers = all_answers[i]

        retrieved_pdf_pages = []
        for result in retrieved_answers:
            source_id = result.get("entity", {}).get("source", {}).get("source_id", "")
            content_metadata = result.get("entity", {}).get("content_metadata", {})
            page_number = content_metadata.get("page_number", "")

            # Extract PDF name from source_id (V2 API normalizes source_id during aggregation,
            # so it should be the original path without #page_X suffixes)
            pdf_name = os.path.basename(source_id).split(".")[0]
            page_str = str(page_number)
            retrieved_pdf_pages.append(f"{pdf_name}_{page_str}")

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
) -> Dict[int, float]:
    """
    Calculate recall@k scores for queries against a Milvus collection using PDF-only matching.

    Matches only PDF filenames (no page numbers), used for datasets like finance_bench where
    ground truth is at document level rather than page level.

    Args:
        query_df: DataFrame with required columns 'query' and 'expected_pdf'.
                  expected_pdf format: PDF filename without extension (e.g., '3M_2018_10K').
        collection_name: Milvus collection name to query.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True. Default False for finance_bench.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        nv_ranker_endpoint: Optional custom reranker endpoint URL.
        nv_ranker_model_name: Optional custom reranker model name.
        ground_truth_dir: Unused parameter (kept for API compatibility).

    Returns:
        Dictionary mapping k values (1, 5, 10) to recall scores (float 0.0-1.0).
        Only includes k values where k <= top_k.
    """
    hits = defaultdict(list)

    # Prepare reranker kwargs if custom endpoint/model provided
    reranker_kwargs = {}
    if nv_ranker:
        if nv_ranker_endpoint:
            reranker_kwargs["nv_ranker_endpoint"] = nv_ranker_endpoint
        if nv_ranker_model_name:
            reranker_kwargs["nv_ranker_model_name"] = nv_ranker_model_name

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
        expected_pdf = query_df["expected_pdf"].iloc[i]
        retrieved_answers = all_answers[i]

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


def bo767_load_ground_truth(ground_truth_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load and organize bo767 ground truth queries from consolidated CSV file.

    Loads the consolidated bo767_query_gt.csv containing all query modalities
    (text, table, chart) and organizes them into separate DataFrames by modality.
    Matches the notebook pattern where queries are filtered by modality value.

    Args:
        ground_truth_dir: Directory containing bo767_query_gt.csv.
                         Defaults to repo data/ directory if None.

    Returns:
        Dictionary mapping modality keys to DataFrames:
        - 'text': Text queries (CSV modality='text')
        - 'tables': Table queries (CSV modality='table', returned as 'tables' for compatibility)
        - 'charts': Chart queries (CSV modality='chart', returned as 'charts' for compatibility)
        - 'multimodal': All queries combined

        Each DataFrame contains columns: query, pdf, page, modality, pdf_page.
        pdf_page format: '{pdf_id}_{page_number}' (1-indexed page numbers).
    """
    if ground_truth_dir is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ground_truth_dir = os.path.join(repo_root, "data")

    csv_path = os.path.join(ground_truth_dir, "bo767_query_gt.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: bo767_query_gt.csv not found at {csv_path}")
        return {}

    df_all = pd.read_csv(csv_path)

    if "pdf" in df_all.columns:
        df_all["pdf"] = df_all["pdf"].astype(str).apply(lambda x: x.replace(".pdf", ""))

    ground_truth = {}

    # Map CSV modality values (singular) to return keys (plural) for backward compatibility
    modality_mapping = [("text", "text"), ("tables", "table"), ("charts", "chart")]
    for modality_key, modality_value in modality_mapping:
        if modality_value in df_all["modality"].values:
            df_modality = df_all[df_all["modality"] == modality_value].copy()
            ground_truth[modality_key] = df_modality.reset_index(drop=True)

    if len(ground_truth) > 0:
        ground_truth["multimodal"] = df_all.copy().reset_index(drop=True)

    return ground_truth


def bo767_recall(
    modalities: List[str],
    collection_names: Dict[str, str],
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate recall@k for bo767 dataset across specified modalities.

    Loads ground truth queries from bo767_query_gt.csv, filters by modality,
    and evaluates recall against the specified Milvus collections.

    Args:
        modalities: List of modalities to evaluate. Valid values: 'multimodal', 'text', 'tables', 'charts'.
        collection_names: Dictionary mapping each modality to its Milvus collection name.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing bo767_query_gt.csv (optional).

    Returns:
        Dictionary mapping modality name to recall scores dictionary.
        Each recall scores dictionary maps k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
    """
    gt = bo767_load_ground_truth(ground_truth_dir)

    results = {}
    for modality in modalities:
        if modality not in collection_names:
            continue

        collection_name = collection_names[modality]

        if modality not in gt:
            print(f"Warning: No ground truth found for modality '{modality}', skipping")
            continue

        query_df = gt[modality]

        scores = get_recall_scores(
            query_df,
            collection_name,
            hostname=hostname,
            sparse=sparse,
            model_name=model_name,
            top_k=top_k,
            gpu_search=gpu_search,
            nv_ranker=nv_ranker,
            ground_truth_dir=ground_truth_dir,
        )

        results[modality] = scores

    return results


def finance_bench_load_ground_truth(ground_truth_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load ground truth query DataFrames for finance_bench dataset.

    Args:
        ground_truth_dir: Directory containing ground truth JSON file.
                         If None, uses repo data/ directory.

    Returns:
        Dictionary with key 'multimodal' containing all queries.
        DataFrame columns: 'query' (from 'question'), 'expected_pdf' (from contexts[0]['filename'])
    """
    if ground_truth_dir is None:
        # Default to repo data/ directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ground_truth_dir = os.path.join(repo_root, "data")

    ground_truth = {}

    # Load finance_bench JSON - expected format: list of dicts with 'question' and 'contexts'
    json_path = os.path.join(ground_truth_dir, "financebench_train.json")
    if not os.path.exists(json_path):
        print(f"Warning: finance_bench ground truth file not found at {json_path}")
        return ground_truth

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

    # Create DataFrame
    if len(queries) > 0:
        df = pd.DataFrame({"query": queries, "expected_pdf": expected_pdfs})
        ground_truth["multimodal"] = df

    return ground_truth


def finance_bench_recall(
    modalities: List[str],
    collection_names: Dict[str, str],
    hostname: str = "localhost",
    sparse: bool = False,  # Default False (hybrid=False) for finance_bench
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate recall@k for finance_bench dataset.

    Loads ground truth queries from financebench_train.json and evaluates recall
    using PDF-only matching (document level, not page level).

    Args:
        modalities: List of modalities to evaluate. Finance_bench uses 'multimodal' only.
        collection_names: Dictionary mapping modality to collection name.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True. Default False for finance_bench.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing financebench_train.json (optional).

    Returns:
        Dictionary mapping modality name to recall scores dictionary.
        Each recall scores dictionary maps k values (1, 5, 10) to recall scores (float 0.0-1.0).
    """
    gt = finance_bench_load_ground_truth(ground_truth_dir)

    if not gt:
        print("Warning: No finance_bench ground truth data loaded")
        return {}

    results = {}
    for modality in modalities:
        if modality not in collection_names:
            continue

        collection_name = collection_names[modality]

        # Finance_bench only has multimodal data
        if modality not in gt:
            print(f"Warning: No ground truth found for modality '{modality}', skipping")
            continue

        query_df = gt[modality]

        # Calculate recall using PDF-only matching
        # Note: finance_bench uses sparse=False (hybrid=False) by default
        scores = get_recall_scores_pdf_only(
            query_df,
            collection_name,
            hostname=hostname,
            sparse=sparse,  # Default False, but can be overridden
            model_name=model_name,
            top_k=top_k,
            gpu_search=gpu_search,
            nv_ranker=nv_ranker,
            ground_truth_dir=ground_truth_dir,
        )

        results[modality] = scores

    return results


def earnings_load_ground_truth(ground_truth_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load ground truth query DataFrames for earnings dataset.

    The earnings CSV uses 0-indexed page numbers (like bo767 text/tables), which are
    converted to 1-indexed in the pdf_page column to match Milvus storage format.

    Args:
        ground_truth_dir: Directory containing ground truth CSV file.
                         If None, uses repo data/ directory.

    Returns:
        Dictionary with keys: individual modality names and 'multimodal'
        DataFrame columns: query, pdf, page, pdf_page
        pdf_page format: '{pdf_id}_{1_indexed_page_number}' (pages converted from 0-indexed CSV)
    """
    if ground_truth_dir is None:
        # Default to repo data/ directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ground_truth_dir = os.path.join(repo_root, "data")

    ground_truth = {}

    # Load earnings CSV - expected format: dir, pdf, page, query, answer, modality
    csv_path = os.path.join(ground_truth_dir, "earnings_consulting_multimodal.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: earnings ground truth file not found at {csv_path}")
        return ground_truth

    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["query", "pdf", "page", "modality"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: earnings CSV missing required columns: {missing_cols}")
        return ground_truth

    # Clean PDF names (remove .pdf extension if present)
    if "pdf" in df.columns:
        df["pdf"] = df["pdf"].apply(lambda x: str(x).replace(".pdf", "") if pd.notna(x) else x)

    # Create pdf_page column: convert 0-indexed CSV pages to 1-indexed for Milvus matching
    # Earnings CSV uses 0-indexed pages (like bo767 text/tables), but Milvus stores 1-indexed
    # Format: '{pdf_id}_{1_indexed_page_number}'
    df["pdf_page"] = df.apply(lambda x: f"{x.pdf}_{x.page + 1}", axis=1)

    # Split by modality
    for modality in df["modality"].unique():
        if pd.isna(modality):
            continue
        modality_df = df[df["modality"] == modality].copy()
        # Keep only necessary columns for recall evaluation
        modality_df = modality_df[["query", "pdf", "page", "pdf_page"]].copy()
        ground_truth[str(modality)] = modality_df.reset_index(drop=True)

    # Create multimodal version (all queries combined)
    if len(df) > 0:
        multimodal_df = df[["query", "pdf", "page", "pdf_page"]].copy()
        ground_truth["multimodal"] = multimodal_df.reset_index(drop=True)

    return ground_truth


def earnings_recall(
    modalities: List[str],
    collection_names: Dict[str, str],
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate recall@k for earnings dataset.

    Loads ground truth queries from earnings_consulting_multimodal.csv and evaluates recall
    using PDF+page matching (matching the old earnings_recall.py pattern).

    Args:
        modalities: List of modalities to evaluate (e.g., ['multimodal', 'text', 'table', 'chart']).
        collection_names: Dictionary mapping modality to collection name.
        hostname: Service hostname for embedding endpoint.
        sparse: Enable hybrid sparse-dense retrieval if True.
        model_name: Embedding model name for query encoding.
        top_k: Maximum number of results to retrieve and evaluate.
        gpu_search: Use GPU acceleration for Milvus search.
        nv_ranker: Enable NVIDIA reranker for result reranking.
        ground_truth_dir: Directory containing earnings_consulting_multimodal.csv (optional).

    Returns:
        Dictionary mapping modality name to recall scores dictionary.
        Each recall scores dictionary maps k values (1, 3, 5, 10) to recall scores (float 0.0-1.0).
    """
    gt = earnings_load_ground_truth(ground_truth_dir)

    if not gt:
        print("Warning: No earnings ground truth data loaded")
        return {}

    results = {}
    for modality in modalities:
        if modality not in collection_names:
            continue

        collection_name = collection_names[modality]

        # Get ground truth for this modality
        if modality not in gt:
            print(f"Warning: No ground truth found for modality '{modality}', skipping")
            continue

        query_df = gt[modality]

        # Calculate recall
        scores = get_recall_scores(
            query_df,
            collection_name,
            hostname=hostname,
            sparse=sparse,
            model_name=model_name,
            top_k=top_k,
            gpu_search=gpu_search,
            nv_ranker=nv_ranker,
            ground_truth_dir=ground_truth_dir,
        )

        results[modality] = scores

    return results


def audio_recall(
    modalities: List[str],
    collection_names: Dict[str, str],
    hostname: str = "localhost",
    sparse: bool = True,
    model_name: str = None,
    top_k: int = 10,
    gpu_search: bool = False,
    nv_ranker: bool = False,
    ground_truth_dir: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    """
    audio dataset recall evaluator (stub).

    TODO: Implement audio-specific ground truth loading and evaluation.
    """
    print("Warning: audio_recall not yet implemented")
    return {}


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
    }

    return evaluators.get(dataset_name.lower())
