"""
Recall evaluation utilities for nv-ingest test cases.

Provides dataset-specific evaluators and helper functions for recall testing.
"""

import os
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
    Calculate recall scores for a given query DataFrame and collection.

    Args:
        query_df: DataFrame with columns 'query' and 'pdf_page'
        collection_name: Milvus collection name
        hostname: Service hostname
        sparse: Whether to use sparse embeddings
        model_name: Embedding model name
        top_k: Maximum k for recall calculation
        gpu_search: Whether to use GPU for search
        nv_ranker: Whether to use NVIDIA reranker
        ground_truth_dir: Directory containing ground truth files (if needed)

    Returns:
        Dictionary mapping k values to recall scores
    """
    hits = defaultdict(list)

    # Run retrieval
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

    # Calculate hits for each query
    for i in range(len(query_df)):
        expected_pdf_page = query_df["pdf_page"].iloc[i]
        retrieved_answers = all_answers[i]

        # Extract pdf_page from retrieved results
        retrieved_pdf_pages = []
        for result in retrieved_answers:
            source_id = result.get("entity", {}).get("source", {}).get("source_id", "")
            content_metadata = result.get("entity", {}).get("content_metadata", {})
            page_number = content_metadata.get("page_number", "")

            # Extract PDF name from source_id (remove .pdf extension if present)
            pdf_name = os.path.basename(source_id).split(".")[0]
            # Use page_number directly as string
            # (matching notebook: str(result['entity']['content_metadata']['page_number']))
            # Milvus stores page_number in 1-indexed format, so no conversion needed
            page_str = str(page_number)

            retrieved_pdf_pages.append(f"{pdf_name}_{page_str}")

        # Check if expected page is in top-k results
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                hits[k].append(expected_pdf_page in retrieved_pdf_pages[:k])

    # Calculate recall scores
    recall_scores = {}
    for k in hits:
        if len(hits[k]) > 0:
            recall_scores[k] = np.mean(hits[k])

    return recall_scores


def bo767_load_ground_truth(ground_truth_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load ground truth query DataFrames for bo767 dataset.

    Args:
        ground_truth_dir: Directory containing ground truth CSV files.
                         If None, uses repo data/ directory.

    Returns:
        Dictionary with keys: 'text', 'tables', 'charts', 'multimodal'
    """
    if ground_truth_dir is None:
        # Default to repo data/ directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ground_truth_dir = os.path.join(repo_root, "data")

    ground_truth = {}

    # Load text queries
    text_path = os.path.join(ground_truth_dir, "text_query_answer_gt_page.csv")
    if os.path.exists(text_path):
        df_text = pd.read_csv(text_path)
        df_text["pdf"] = df_text["pdf"].apply(lambda x: x.replace(".pdf", ""))
        # Note: gt_page is 0-indexed, convert to 1-indexed to match notebook format
        df_text["pdf_page"] = df_text.apply(lambda x: f"{x.pdf}_{x.gt_page + 1}", axis=1)
        ground_truth["text"] = df_text

    # Load table queries
    table_path = os.path.join(ground_truth_dir, "table_queries_cleaned_235.csv")
    if os.path.exists(table_path):
        df_table = pd.read_csv(table_path)[["query", "pdf", "page", "table"]]
        # Note: page is 0-indexed, convert to 1-indexed to match notebook format
        df_table["pdf_page"] = df_table.apply(lambda x: f"{x.pdf}_{x.page + 1}", axis=1)
        ground_truth["tables"] = df_table

    # Load chart queries
    chart_path = os.path.join(ground_truth_dir, "charts_with_page_num_fixed.csv")
    if os.path.exists(chart_path):
        df_chart = pd.read_csv(chart_path)[["query", "pdf", "page"]]
        # Note: chart CSV already has 1-indexed pages, use directly
        df_chart["pdf_page"] = df_chart.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
        ground_truth["charts"] = df_chart

    # Create multimodal queries (combination of all)
    if "text" in ground_truth and "tables" in ground_truth and "charts" in ground_truth:
        # Text: gt_page is 0-indexed, keep as 0-indexed (will add 1 in pdf_page formula)
        df_multimodal = ground_truth["text"].copy()
        df_multimodal["page"] = df_multimodal["gt_page"]  # Keep 0-indexed
        df_multimodal = df_multimodal[["query", "pdf", "page"]].copy()
        df_multimodal["modality"] = "text"

        # Tables: page is 0-indexed, keep as 0-indexed (will add 1 in pdf_page formula)
        df_table_mod = ground_truth["tables"].copy()
        # page is already 0-indexed, no conversion needed
        df_table_mod = df_table_mod[["query", "pdf", "page"]].copy()
        df_table_mod["modality"] = "table"

        # Charts: CSV has 1-indexed pages, convert to 0-indexed (matching notebook logic)
        df_chart_mod = ground_truth["charts"][["query", "pdf", "page"]].copy()
        df_chart_mod["page"] = df_chart_mod["page"] - 1  # Convert 1-indexed to 0-indexed
        df_chart_mod["modality"] = "chart"

        df_multimodal = pd.concat([df_multimodal, df_table_mod, df_chart_mod]).reset_index(drop=True)
        # All pages are now 0-indexed, add 1 to convert to 1-indexed in pdf_page (matching notebook)
        df_multimodal["pdf_page"] = df_multimodal.apply(lambda x: f"{x.pdf}_{x.page + 1}", axis=1)
        ground_truth["multimodal"] = df_multimodal

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
    bo767 dataset recall evaluator.

    Args:
        modalities: List of modalities to evaluate (e.g., ['multimodal', 'text'])
        collection_names: Dictionary mapping modality to collection name
        hostname: Service hostname
        sparse: Whether to use sparse embeddings
        model_name: Embedding model name
        top_k: Maximum k for recall calculation
        gpu_search: Whether to use GPU for search
        nv_ranker: Whether to use NVIDIA reranker
        ground_truth_dir: Directory containing ground truth CSV files

    Returns:
        Dictionary mapping modality to recall scores (k -> score)
    """
    gt = bo767_load_ground_truth(ground_truth_dir)

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


def finance_bench_recall(
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
    finance_bench dataset recall evaluator (stub).

    TODO: Implement finance_bench-specific ground truth loading and evaluation.
    """
    # Stub implementation
    print("Warning: finance_bench_recall not yet implemented")
    return {}


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
    earnings dataset recall evaluator (stub).

    TODO: Implement earnings-specific ground truth loading and evaluation.
    """
    # Stub implementation
    print("Warning: earnings_recall not yet implemented")
    return {}


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
