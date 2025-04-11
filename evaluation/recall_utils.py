from nv_ingest_client.util.milvus import nvingest_retrieval
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import pickle


def get_recall_scores(query_df, collection_name, hybrid):
    hits = defaultdict(list)
    all_answers = nvingest_retrieval(
        query_df["query"].to_list(),
        collection_name,
        hybrid=hybrid,
        embedding_endpoint="http://localhost:8012/v1",
        model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        top_k=10,
    )

    results_df = pd.DataFrame()

    for i in range(len(query_df)):
        expected_pdf_page = query_df["pdf_page"][i]
        retrieved_answers = all_answers[i]
        retrieved_pdfs = [
            os.path.basename(result["entity"]["source"]["source_id"]).split(".")[0] for result in retrieved_answers
        ]
        retrieved_pages = [str(result["entity"]["content_metadata"]["page_number"]) for result in retrieved_answers]
        retrieved_pdf_pages = [f"{pdf}_{page}" for pdf, page in zip(retrieved_pdfs, retrieved_pages)]

        result = {
            "query_id": query_df["query_id"][i],
            "query_text": query_df["query"][i],
            "expected_pdf_page": expected_pdf_page,
            "expected_answer": query_df["answer"][i],
            "retrieved_pdf_pages": [retrieved_pdf_pages],
            "retrieved_pdf_texts": [[result["entity"]["text"] for result in retrieved_answers]],
        }
        results_df = pd.concat([results_df, pd.DataFrame(result, index=[0])], ignore_index=True)

        for k in [1, 5, 10]:
            hits[k].append(expected_pdf_page in retrieved_pdf_pages[:k])

    for k in hits:
        print(f"  - Recall @{k}: {np.mean(hits[k]) :.3f}")

    return results_df


def get_bo_results():
    with open('earnings_results_bo.pkl', 'rb') as f:
        df = pickle.load(f)
        return df
