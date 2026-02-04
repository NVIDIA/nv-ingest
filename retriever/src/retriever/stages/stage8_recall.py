from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm
import glob
import pandas as pd
import os
import torch.nn.functional as F

import llama_nemotron_embed_1b_v2

# from slimgest.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
# import slimgest.model.local.llama_nemotron_embed_1b_v2_embedder as llama_nemotron_embed_1b_v2
from slimgest.local.vdb.lancedb import LanceDB
import lancedb


install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 7: load results from stage6, upload embeddings to VDB"
)

DEFAULT_INPUT_DIR = Path("./data/pages")


def calculate_recall(real_answers, retrieved_answers, k):
    hits = 0
    for real, retrieved in zip(real_answers, retrieved_answers):
        if real in retrieved[:k]:
            hits += 1
    return hits / len(real_answers)


def calcuate_recall_list(real_answers, retrieved_answers, ks=[1, 3, 5, 10]):
    recall_scores = {}
    for k in ks:
        recall_scores[k] = calculate_recall(real_answers, retrieved_answers, k)
    return recall_scores

def analyze_query_hits(real_answers, retrieved_answers, query_texts, ks=[1, 3, 5, 10]):
    """
    Analyze individual query hits and track which queries succeeded/failed.
    
    Returns:
        results_df: DataFrame with detailed per-query results
        hit_summary: Dictionary with hit/miss counts for each k
    """
    results = []
    hit_summary = {k: {"hits": 0, "misses": 0} for k in ks}
    
    for idx, (real, retrieved, query) in enumerate(zip(real_answers, retrieved_answers, query_texts)):
        query_result = {
            "query_id": idx,
            "query": query,
            "golden_answer": real,
            "top_10_retrieved": retrieved[:10],
        }
        
        # Check if hit at each k value
        for k in ks:
            hit = real in retrieved[:k]
            query_result[f"hit@{k}"] = hit
            if hit:
                hit_summary[k]["hits"] += 1
                # Find the rank where it was found
                try:
                    rank = retrieved[:k].index(real) + 1
                    query_result[f"rank@{k}"] = rank
                except ValueError:
                    query_result[f"rank@{k}"] = None
            else:
                hit_summary[k]["misses"] += 1
                query_result[f"rank@{k}"] = None
        
        results.append(query_result)
    
    results_df = pd.DataFrame(results)
    return results_df, hit_summary

def get_correct_answers(query_df):
    retrieved_pdf_pages = []
    for i in range(len(query_df)):
        retrieved_pdf_pages.append(query_df['pdf_page'][i]) 
    return retrieved_pdf_pages

def create_lancedb_results(results):
    old_results = [res["metadata"] for result in results for res in result]
    results = []
    for result in old_results:
        if result["embedding"] is None:
            continue
        results.append({
            "vector": result["embedding"], 
            "text": result["content"], 
            "metadata": result["content_metadata"]["page_number"], 
            "source": result["source_metadata"]["source_id"],
        })
    return results

def format_retrieved_answers_lance(all_answers):
    retrieved_pdf_pages = []
    for answers in all_answers:
        retrieved_pdfs = [os.path.basename(result['source']).split('_')[0] for result in answers]
        retrieved_pages = [str(result['metadata']) for result in answers]
        retrieved_pdf_pages.append([f"{pdf}_{page}" for pdf, page in zip(retrieved_pdfs, retrieved_pages)])
    return retrieved_pdf_pages



def average_pool(last_hidden_states, attention_mask):
    """Average pooling with attention mask."""
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    return embedding



@app.command()
def run(
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model."),
    query_file: Path = typer.Option(..., "--query-file", exists=True, file_okay=True, dir_okay=False, help="File with one query per line."),
    hf_cache_dir: Path = typer.Option(
        Path.home() / ".cache" / "huggingface",
        "--hf-cache-dir",
        help="Local cache dir for large model files (e.g. model.safetensors).",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/recall_analysis"),
        "--output-dir",
        help="Directory to save detailed results and analysis.",
    ),
):
    processed = 0
    skipped = 0
    # Ensure the cache directory exists so large weights are reused across runs.
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: force_download=True will re-download huge weights every run. Keep it False.
    embed_model = llama_nemotron_embed_1b_v2.load_model(
        device=device,
        trust_remote_code=True,
        cache_dir=str(hf_cache_dir),
        force_download=False,
    )
    tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer(
        cache_dir=str(hf_cache_dir),
        force_download=False,
    )

    db = lancedb.connect(uri="lancedb")
    table = db.open_table("nv-ingest")

    # Use the shared embedder wrapper; if endpoint is set, it runs remotely.
    df_query = pd.read_csv(query_file).rename(columns={'gt_page':'page'})[['query','pdf','page']]
    df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1) 

    console.print(f"[bold cyan]Stage8[/bold cyan] images={df_query.shape[0]} input_dir={query_file} device={device} hf_cache_dir={hf_cache_dir}")


    all_answers = []
    query_texts = df_query['query'].tolist()
    top_k = 10
    result_fields = ["source", "metadata", "text", "_distance"]
    query_embeddings = []
    for query_batch in tqdm(query_texts, desc="Stage8 query embeddings", unit="queries"):
        tokenized_inputs = tokenizer(["query: " + query_batch], return_tensors="pt", padding=True, truncation=True).to(device)
        embed_model_results = embed_model(**tokenized_inputs)
        emb = average_pool(embed_model_results.last_hidden_state, tokenized_inputs['attention_mask']).detach().cpu().numpy()
        query_embeddings.append(emb)

    for query_embed in tqdm(query_embeddings, desc="Stage8 querying VDB", unit="queries"): 
        all_answers.append(
            table.search(query_embed, vector_column_name="vector").select(result_fields).limit(top_k).to_list()
        )
        processed += 1
    retrieved_pdf_pages = format_retrieved_answers_lance(all_answers)
    golden_answers = get_correct_answers(df_query)
    recall_scores = calcuate_recall_list(golden_answers, retrieved_pdf_pages, ks=[1, 3, 5, 10])
    
    # Analyze individual query hits
    results_df, hit_summary = analyze_query_hits(
        golden_answers, 
        retrieved_pdf_pages, 
        query_texts, 
        ks=[1, 3, 5, 10]
    )
    
    # Save detailed results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"recall_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    console.print(f"[blue]Saved detailed results to:[/blue] {results_file}")
    
    # Save separate files for hits and misses at different k values
    for k in [1, 3, 5, 10]:
        hits_df = results_df[results_df[f"hit@{k}"] == True]
        misses_df = results_df[results_df[f"hit@{k}"] == False]
        
        hits_file = output_dir / f"hits_at_{k}_{timestamp}.csv"
        misses_file = output_dir / f"misses_at_{k}_{timestamp}.csv"
        
        hits_df.to_csv(hits_file, index=False)
        misses_df.to_csv(misses_file, index=False)
        
        console.print(
            f"[green]k={k}:[/green] {hit_summary[k]['hits']} hits, {hit_summary[k]['misses']} misses "
            f"(saved to hits_at_{k}.csv and misses_at_{k}.csv)"
        )

    console.print(
        f"\n[green]Done[/green] processed={processed} skipped={skipped} "
        f"recall@1={recall_scores[1]:.4f} recall@3={recall_scores[3]:.4f} "
        f"recall@5={recall_scores[5]:.4f} recall@10={recall_scores[10]:.4f}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

