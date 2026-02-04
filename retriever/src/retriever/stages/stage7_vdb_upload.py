from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm
import glob

from slimgest.local.vdb.lancedb import LanceDB


install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 7: load results from stage6, upload embeddings to VDB"
)

DEFAULT_INPUT_DIR = Path("./data/pages")


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model."),

):
    # Use the shared embedder wrapper; if endpoint is set, it runs remotely.

    files_to_read = glob.glob(str(input_dir / "*.embeddings.pt"))

    console.print(f"[bold cyan]Stage7[/bold cyan] images={len(files_to_read)} input_dir={input_dir} device={device}")

    processed = 0
    skipped = 0
    missing_stage5 = 0
    bad_stage5 = 0
    missing_pdfium_text = 0
    full_chunks = []
    chunks_appended = 0
    for file_path in tqdm(files_to_read, desc="Stage7 creating_chunks", unit="pt file"):
        # create chunks
        with open(file_path, "rb") as f:
            data = torch.load(f, weights_only=False)
        for emb, text in zip(data["embeddings"], data["texts"]):
            if emb is not None and text:
                chunk = {
                    "embedding": emb.numpy() if hasattr(emb, "numpy") else emb,
                    "content": text,
                    "source_id": file_path,
                    "page_number": int(file_path.split("_page")[-1].split(".")[0]),
                }
                full_chunks.append(chunk)
                chunks_appended += 1
            else:
                skipped += 1

    # upload to vdb

    vdb = LanceDB(overwrite=True)
    vdb.run(full_chunks)

    chunks_written_to_lancedb = vdb.table.count_rows()

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_stage5={missing_stage5} bad_stage5={bad_stage5} "
        f"missing_pdfium_text={missing_pdfium_text} wrote_pt_suffix=.embeddings.pt"
    )
    console.print(
        f"[bold yellow]Chunk Stats[/bold yellow] chunks_appended_to_full_chunks={chunks_appended} "
        f"chunks_written_to_lancedb={chunks_written_to_lancedb}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

