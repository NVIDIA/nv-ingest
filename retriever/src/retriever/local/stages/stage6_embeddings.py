from __future__ import annotations

import csv
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm

from ._io import coerce_embedding_to_vector, iter_images, normalize_l2, read_json

# import slimgest.model.local.llama_nemotron_embed_1b_v2_embedder as llama_nemotron_embed_1b_v2
import numpy as np

# from slimgest.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
import llama_nemotron_embed_1b_v2
import torch.nn.functional as F

install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 6: load OCR JSON from stage5, embed each OCR region, and save a torch-loadable .pt alongside each image."
)

DEFAULT_INPUT_DIR = Path("./data/pages")


def average_pool(last_hidden_states, attention_mask):
    """Average pooling with attention mask."""
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    return embedding


def _stage5_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".nemotron_ocr_v1.json")


def _pdfium_text_for_image(img_path: Path) -> Path:
    """
    pdf.convert writes embedded PDF text as:
      <pdf_basename>_page<NNNN>.pdfium_text.txt

    For an image like:
      <pdf_basename>_page<NNNN>.png

    ...this corresponds to `img_path.with_suffix(".pdfium_text.txt")`.
    """
    return img_path.with_suffix(".pdfium_text.txt")


def _read_text_file_best_effort(path: Path) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    # Keep internal newlines intact; just trim surrounding whitespace.
    return s.strip()


def _embed_via_remote_endpoint(
    texts: List[str],
    endpoint_url: str,
    model_name: Optional[str] = None,
    batch_size: int = 64,
) -> List[np.ndarray]:
    """
    Embed texts using a remote OpenAI-compatible embedding endpoint.
    Returns list of numpy arrays (one per text).

    Configured for NVIDIA NeMo Retriever Llama 3.2 embedding model:
    - Max context length: 8192 tokens
    - Embedding dimensions: 384, 512, 768, 1024, or 2048 (default: 2048)
    - Input type: "passage" for documents, "query" for questions
    """
    all_embeddings: List[np.ndarray] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Build payload - don't use extra_body, put parameters directly in the request
        payload = {
            "input": batch,
            "encoding_format": "float",
            "input_type": "passage",  # For embedding documents/passages
            "truncate": "NONE",
        }
        if model_name:
            payload["model"] = model_name

        try:
            response = requests.post(
                endpoint_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300,  # 5 minute timeout
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Include response body in error message for debugging
            error_detail = ""
            try:
                error_detail = f" Response: {response.text}"
            except Exception:
                pass
            raise requests.exceptions.HTTPError(f"{str(e)}{error_detail}", response=response)

        result = response.json()

        # Extract embeddings from response
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        embeddings_data = result.get("data", [])
        # Sort by index to ensure correct order
        embeddings_data = sorted(embeddings_data, key=lambda x: x.get("index", 0))

        for item in embeddings_data:
            embedding = item.get("embedding", [])
            all_embeddings.append(np.array(embedding, dtype=np.float32))

    return all_embeddings


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".embeddings.pt")


def _embedder_input_path_for_image(img_path: Path) -> Path:
    # Intentionally uses a "double extension" so it's easy to spot in a pages dir.
    # Example: foo_page0001.png.embedder-input.txt
    return img_path.with_name(img_path.name + ".embedder-input.txt")


def _write_embedder_input_txt(
    path: Path,
    *,
    embedder_texts: List[str],
) -> None:
    """
    Write only the exact text fed into the embedder.

    The embedder receives a list of strings; this file writes them in order, with a
    single newline between entries (no extra headers/metadata).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(embedder_texts).rstrip() + "\n", encoding="utf-8", errors="replace")


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .pt outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
    hf_cache_dir: Path = typer.Option(
        Path.home() / ".cache" / "huggingface",
        "--hf-cache-dir",
        help="Local cache dir for large model files (e.g. model.safetensors).",
    ),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        help="Optional embedding endpoint URL (often OpenAI-compatible '/v1/embeddings'). If set, embeddings run remotely and the local embedding model is not loaded.",
    ),
    embedding_model_name: Optional[str] = typer.Option(
        None,
        help="Optional embedding model name for remote endpoint (sent as 'model' in payload).",
    ),
    batch_size: int = typer.Option(64, "--batch-size", min=1, help="Embedding batch size (texts per request)."),
    write_embedder_input: bool = typer.Option(
        True,
        "--write-embedder-input/--no-write-embedder-input",
        help="Write a per-page .embedder-input.txt file showing the exact text fed to the embedder.",
    ),
):
    dev = torch.device(device)
    # Ensure the cache directory exists so large weights are reused across runs.
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # Only load local embedding model if not using a remote endpoint
    embed_model = None
    tokenizer = None
    if not embedding_endpoint:
        console.print("[cyan]Loading local embedding model...[/cyan]")
        # IMPORTANT: force_download=True will re-download huge weights every run. Keep it False.
        embed_model = llama_nemotron_embed_1b_v2.load_model(
            device=str(dev),
            trust_remote_code=True,
            cache_dir=str(hf_cache_dir),
            force_download=False,
        )
        tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer(
            cache_dir=str(hf_cache_dir),
            force_download=False,
        )
    else:
        console.print(f"[cyan]Using remote embedding endpoint: {embedding_endpoint}[/cyan]")
        if embedding_model_name:
            console.print(f"[cyan]Model name: {embedding_model_name}[/cyan]")

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(
        f"[bold cyan]Stage6[/bold cyan] images={len(images)} input_dir={input_dir} device={dev} hf_cache_dir={hf_cache_dir}"
    )

    # Track all skipped items with reasons
    skip_records: List[Dict[str, str]] = []

    chunks_created = 0
    processed = 0
    skipped = 0
    oom_errors = 0
    remote_endpoint_errors = 0
    missing_stage5 = 0
    bad_stage5 = 0
    missing_pdfium_text = 0
    texts_added = 0
    ocr_added = 0
    for img_path in tqdm(images, desc=f"Stage6 images", unit="img"):
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            skip_records.append(
                {
                    "image_path": str(img_path),
                    "reason": "already_exists",
                    "details": f"Output file {out_path.name} already exists and --overwrite not set",
                }
            )
            continue

        s5_path = _stage5_json_for_image(img_path)
        pdfium_text_path = _pdfium_text_for_image(img_path)
        pdfium_text = ""

        # Read pdfium text if available
        if pdfium_text_path.exists():
            pdfium_text = _read_text_file_best_effort(pdfium_text_path)
        else:
            missing_pdfium_text += 1

        # Try to read stage5 results, but allow processing to continue without them
        regions: List[Dict[str, Any]] = []
        stage5_status = ""
        if s5_path.exists():
            try:
                s5 = read_json(s5_path)
                regions = list(s5.get("regions") or [])
                stage5_status = "found"
            except Exception as e:
                bad_stage5 += 1
                stage5_status = f"error: {str(e)}"
                # Continue with empty regions - we may still have pdfium_text
        else:
            missing_stage5 += 1
            stage5_status = "missing"
            # Continue with empty regions - we may still have pdfium_text

        # If we have neither stage5 regions nor pdfium_text, skip this image
        if not regions and not pdfium_text:
            skipped += 1
            skip_records.append(
                {
                    "image_path": str(img_path),
                    "reason": "no_content",
                    "details": f"No stage5 regions ({stage5_status}) and no pdfium_text",
                }
            )
            continue
        texts: List[str] = []
        bboxes: List[List[float]] = []
        text_kinds: List[str] = []

        # Include embedded PDF text (if present) as a full-page "region" so it is embedded
        # alongside OCR detection text.
        if pdfium_text:
            texts.append(pdfium_text)
            bboxes.append([0.0, 0.0, 1.0, 1.0])
            text_kinds.append("pdfium_page_text")
            texts_added += 1

        for r in regions:
            txt = (r.get("ocr_text") or "").strip()
            bbox = r.get("bbox_xyxy_norm_in_page")
            if txt:
                texts.append(txt)
                if isinstance(bbox, list) and len(bbox) == 4:
                    bboxes.append([float(x) for x in bbox])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
                text_kinds.append("ocr_region")
                ocr_added += 1

        t0 = time.perf_counter()
        vectors: List[np.ndarray] = []
        try:
            if texts:
                raw_texts = list(texts)

                if embedding_endpoint:
                    # Use remote endpoint for embeddings (don't prepend "passage: ")
                    # The input_type is specified in the API call's extra_body parameter
                    embedder_texts = raw_texts
                    vectors = _embed_via_remote_endpoint(
                        embedder_texts,
                        embedding_endpoint,
                        model_name=embedding_model_name,
                        batch_size=batch_size,
                    )
                else:
                    # Use local model for embeddings (prepend "passage: " for local model)
                    embedder_texts = ["passage: " + text for text in raw_texts]
                    with torch.inference_mode():
                        tokenized_inputs = tokenizer(
                            embedder_texts, return_tensors="pt", padding=True, truncation=True
                        ).to(dev)
                        embed_model_results = embed_model(**tokenized_inputs)
                        emb = average_pool(embed_model_results.last_hidden_state, tokenized_inputs["attention_mask"])

                        for i in range(int(emb.shape[0])):
                            vectors.append(emb[i].cpu().numpy().astype(np.float32))

                # Write embedder input for debugging/inspection
                if write_embedder_input:
                    _write_embedder_input_txt(
                        _embedder_input_path_for_image(img_path),
                        embedder_texts=embedder_texts,
                    )
        except requests.exceptions.RequestException as e:
            # Handle remote endpoint errors
            remote_endpoint_errors += 1
            skip_records.append(
                {
                    "image_path": str(img_path),
                    "reason": "remote_endpoint_error",
                    "details": f"Remote endpoint error: {str(e)}",
                }
            )
            console.print(
                f"[bold red]Remote Endpoint Error[/bold red] while embedding {img_path} "
                f"(skipping this page and continuing): {e}"
            )
            continue
        except torch.cuda.OutOfMemoryError as e:
            oom_errors += 1
            skip_records.append(
                {"image_path": str(img_path), "reason": "cuda_oom", "details": f"CUDA OutOfMemoryError: {str(e)}"}
            )
            console.print(
                f"[bold red]CUDA OOM[/bold red] while embedding {img_path} " f"(skipping this page and continuing): {e}"
            )
            # Best-effort cleanup so we can continue.
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        except RuntimeError as e:
            # Some CUDA OOMs surface as generic RuntimeError("CUDA out of memory ...").
            msg = str(e)
            if "CUDA out of memory" in msg or "cuda out of memory" in msg:
                oom_errors += 1
                skip_records.append(
                    {
                        "image_path": str(img_path),
                        "reason": "cuda_oom_runtime",
                        "details": f"RuntimeError CUDA OOM: {str(e)}",
                    }
                )
                console.print(
                    f"[bold red]CUDA OOM[/bold red] while embedding {img_path} "
                    f"(skipping this page and continuing): {e}"
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
        finally:
            # Release any large tensors promptly between pages.
            try:
                del tokenized_inputs  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                del embed_model_results  # type: ignore[name-defined]
            except Exception:
                pass
            try:
                del emb  # type: ignore[name-defined]
            except Exception:
                pass

        dt = time.perf_counter() - t0
        chunks_created += len(texts)

        torch.save(
            {
                "schema_version": 1,
                "stage": 6,
                "model": "llama_nemotron_embed_1b_v2",
                "image_path": str(img_path),
                "stage5_json": str(s5_path),
                "pdfium_text_path": str(pdfium_text_path),
                "texts": texts,
                "text_kinds": text_kinds,
                "bboxes_xyxy_norm_in_page": bboxes,
                "embeddings": vectors,
                "timing": {"seconds": float(dt)},
            },
            out_path,
        )
        processed += 1

    # Write skip records to CSV file
    if skip_records:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skip_log_path = input_dir / f"stage6_skipped_{timestamp}.csv"
        try:
            with open(skip_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "reason", "details"])
                writer.writeheader()
                writer.writerows(skip_records)
            console.print(f"[yellow]Wrote skip log to:[/yellow] {skip_log_path} ({len(skip_records)} items)")
        except Exception as e:
            console.print(f"[red]Warning: Failed to write skip log:[/red] {e}")

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_stage5={missing_stage5} bad_stage5={bad_stage5} "
        f"missing_pdfium_text={missing_pdfium_text} oom_errors={oom_errors} remote_endpoint_errors={remote_endpoint_errors} "
        f"chunks_created={chunks_created} texts_added={texts_added} ocr_added={ocr_added} wrote_pt_suffix=.embeddings.pt"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
