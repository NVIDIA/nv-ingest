from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table
from rich.traceback import install
import torch
import typer
from tqdm import tqdm
from PIL import Image

from slimgest.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3

from ._io import PAGE_ELEMENT_LABELS, IMAGE_EXTS, iter_images, load_image_rgb_chw_u8, to_bbox_list, to_scalar_float, to_scalar_int, write_json


install(show_locals=False)
console = Console()
app = typer.Typer(help="Stage 2: run page_elements_v3 over page images and save JSON alongside each image.")

# Configure your checkpoint directory here (can still override via CLI).
DEFAULT_INPUT_DIR = Path("./data/pages")


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".page_elements_v3.json")


def _chunked(seq: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def _invoke_batched_page_elements(
    model: NemotronPageElementsV3,
    batch_tensor: torch.Tensor,
    orig_shapes: Sequence[Tuple[int, int]],
) -> List[Any]:
    """
    Best-effort batched inference.

    Underlying Nemotron implementations vary; some accept a list of shapes, some accept a single shape,
    and some do not support batching at all. This function normalizes successful outputs into a
    list of per-image prediction objects (length == batch size), or raises to allow fallback.
    """
    # If using remote endpoint, bypass local nn.Module batching and use remote batch API.
    if getattr(model, "_endpoint", None) is not None:
        out = model.invoke_remote(batch_tensor)
        if isinstance(out, list) and len(out) == int(batch_tensor.shape[0]):
            return out
        raise RuntimeError("Remote output was not per-image list; falling back to per-image inference.")

    m = model.model  # underlying callable
    # Try (BCHW, list[shape]) first, then (BCHW, shape)
    try:
        raw = m(batch_tensor, list(orig_shapes))
    except Exception:
        raw = m(batch_tensor, orig_shapes[0])

    # Many implementations return (preds, aux...), where preds is a per-image list.
    raw0 = raw[0] if isinstance(raw, (tuple, list)) and len(raw) > 0 else raw
    if isinstance(raw0, list) and len(raw0) == int(batch_tensor.shape[0]):
        return raw0
    raise RuntimeError("Batched model output was not per-image list; falling back to per-image inference.")


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for model + tensors."),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Best-effort inference batch size."),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Optional page elements NIM endpoint URL. If set, runs remotely and local weights are not loaded.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
):
    """
    Reads images from input_dir (stage 1 outputs) and writes <image>.+page_elements_v3.json.
    """
    dev = torch.device(device)
    model = NemotronPageElementsV3(endpoint=endpoint, remote_batch_size=batch_size)

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(f"[bold cyan]Stage2[/bold cyan] images={len(images)} input_dir={input_dir} device={dev}")

    processed = 0
    skipped = 0
    to_process: List[Path] = []
    for img_path in images:
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        to_process.append(img_path)

    batches = list(_chunked(to_process, batch_size))
    images_done = 0
    pbar = tqdm(batches, desc="Stage2", unit="batch")
    pbar.set_postfix(img=f"{images_done}/{len(to_process)}")
    for batch_paths in pbar:
        # Load + preprocess
        tensors: List[torch.Tensor] = []
        shapes: List[Tuple[int, int]] = []
        for p in batch_paths:
            t, (h, w) = load_image_rgb_chw_u8(p, dev)
            shapes.append((h, w))
            tensors.append(model.preprocess(t))

        t0 = time.perf_counter()
        with torch.inference_mode():
            # Attempt true batching; if unavailable, fall back to per-image invoke
            per_image_preds: Optional[List[Any]] = None
            try:
                batch_tensor = torch.stack(tensors, dim=0)
                per_image_preds = _invoke_batched_page_elements(model, batch_tensor, shapes)
            except Exception:
                per_image_preds = None

            for i, img_path in enumerate(batch_paths):
                orig_shape = shapes[i]
                if per_image_preds is None:
                    preds = model.invoke(tensors[i], orig_shape)
                else:
                    preds = per_image_preds[i]

                boxes, labels, scores = model.postprocess(preds)
                dets: List[Dict[str, Any]] = []
                for box, lab, score in zip(boxes, labels, scores):
                    lab_i = to_scalar_int(lab)
                    label_name = PAGE_ELEMENT_LABELS.get(int(lab_i), f"unknown_{int(lab_i)}")
                    dets.append(
                        {
                            "bbox_xyxy_norm": to_bbox_list(box),
                            "label": lab_i,
                            "label_name": label_name,
                            "score": to_scalar_float(score),
                        }
                    )

                dt = time.perf_counter() - t0
                payload: Dict[str, Any] = {
                    "schema_version": 1,
                    "stage": 2,
                    "model": "page_elements_v3",
                    "image": {
                        "path": str(img_path),
                        "height": int(orig_shape[0]),
                        "width": int(orig_shape[1]),
                    },
                    "detections": dets,
                    "timing": {"seconds": dt},
                }
                out_path = _out_path_for_image(img_path)
                write_json(out_path, payload)
                processed += 1

        images_done += len(batch_paths)
        pbar.set_postfix(img=f"{images_done}/{len(to_process)}", batch_imgs=str(len(batch_paths)))

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} wrote_json_suffix=.page_elements_v3.json"
    )


@app.command()
def validate_input(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    expected_width: Optional[int] = typer.Option(None, "--expected-width", help="Expected image width in pixels."),
    expected_height: Optional[int] = typer.Option(None, "--expected-height", help="Expected image height in pixels."),
    expected_count: Optional[int] = typer.Option(None, "--expected-count", help="Expected number of image files."),
):
    """
    Validate all images in input_dir: check format, dimensions, and count.
    
    This command scans the input directory and generates a validation report showing:
    - Total number of image files found
    - Total bytes of all image files
    - List of non-compliant files (wrong format, wrong dimensions)
    """
    console.print(f"[bold cyan]Validating inputs[/bold cyan] in {input_dir}")
    
    # Collect all files in the directory
    all_files = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    
    # Separate image files from non-image files
    image_files = iter_images(input_dir)
    non_image_files = [p for p in all_files if p not in image_files]
    
    # Track statistics
    total_bytes = 0
    non_compliant_files: List[Dict[str, Any]] = []
    dimension_stats: Dict[Tuple[int, int], int] = {}
    format_stats: Dict[str, int] = {}
    compliant_count = 0
    missing_text_files = 0
    empty_text_files = 0
    missing_image_only_files = 0
    empty_image_only_files = 0
    
    # Create progress bar with detailed information
    pbar = tqdm(image_files, desc="Validating", unit="file")
    
    for img_path in pbar:
        file_size = img_path.stat().st_size
        total_bytes += file_size
        is_compliant = True
        
        # Update progress bar with current file info
        file_name = img_path.name[:40] + "..." if len(img_path.name) > 40 else img_path.name
        pbar.set_postfix({
            "current": file_name,
            "compliant": compliant_count,
            "issues": len(non_compliant_files),
        })
        
        try:
            # Check if extension is valid
            ext = img_path.suffix.lower()
            if ext not in IMAGE_EXTS:
                non_compliant_files.append({
                    "path": str(img_path.relative_to(input_dir)),
                    "reason": f"Invalid extension: {ext}",
                    "size_bytes": file_size,
                })
                is_compliant = False
                continue
            
            # Track format statistics
            format_stats[ext] = format_stats.get(ext, 0) + 1
            
            # Check for corresponding pdfium_text.txt file
            text_file_path = img_path.with_name(img_path.name + ".pdfium_text.txt")
            if not text_file_path.exists():
                non_compliant_files.append({
                    "path": str(img_path.relative_to(input_dir)),
                    "reason": "Missing pdfium_text.txt file",
                    "size_bytes": file_size,
                    "text_file": str(text_file_path.name),
                })
                missing_text_files += 1
                is_compliant = False
            else:
                # Check if text file is empty
                text_file_size = text_file_path.stat().st_size
                if text_file_size == 0:
                    non_compliant_files.append({
                        "path": str(img_path.relative_to(input_dir)),
                        "reason": "Empty pdfium_text.txt file (0 bytes)",
                        "size_bytes": file_size,
                        "text_file": str(text_file_path.name),
                    })
                    empty_text_files += 1
                    is_compliant = False
            
            # Check for corresponding pdfium_image_only.txt file
            image_only_file_path = img_path.with_name(img_path.name + ".pdfium_image_only.txt")
            if not image_only_file_path.exists():
                non_compliant_files.append({
                    "path": str(img_path.relative_to(input_dir)),
                    "reason": "Missing pdfium_image_only.txt file",
                    "size_bytes": file_size,
                    "text_file": str(image_only_file_path.name),
                })
                missing_image_only_files += 1
                is_compliant = False
            else:
                # Check if image_only file is empty
                image_only_file_size = image_only_file_path.stat().st_size
                if image_only_file_size == 0:
                    non_compliant_files.append({
                        "path": str(img_path.relative_to(input_dir)),
                        "reason": "Empty pdfium_image_only.txt file (0 bytes)",
                        "size_bytes": file_size,
                        "text_file": str(image_only_file_path.name),
                    })
                    empty_image_only_files += 1
                    is_compliant = False
            
            # Load and check dimensions
            with Image.open(img_path) as im:
                width, height = im.size
                
                # Track dimension statistics
                dimension_stats[(height, width)] = dimension_stats.get((height, width), 0) + 1
                
                # Check if dimensions match expected values
                dimension_issues = []
                if expected_width is not None and width != expected_width:
                    dimension_issues.append(f"width={width} (expected {expected_width})")
                if expected_height is not None and height != expected_height:
                    dimension_issues.append(f"height={height} (expected {expected_height})")
                
                if dimension_issues:
                    non_compliant_files.append({
                        "path": str(img_path.relative_to(input_dir)),
                        "reason": f"Dimension mismatch: {', '.join(dimension_issues)}",
                        "size_bytes": file_size,
                        "dimensions": f"{height}x{width}",
                    })
                    is_compliant = False
            
            if is_compliant:
                compliant_count += 1
        
        except Exception as e:
            non_compliant_files.append({
                "path": str(img_path.relative_to(input_dir)),
                "reason": f"Error loading image: {str(e)}",
                "size_bytes": file_size,
            })
    
    pbar.close()
    
    # Add non-image files to total bytes
    for p in non_image_files:
        total_bytes += p.stat().st_size
    
    # Print summary report
    console.print("\n" + "=" * 80)
    console.print("[bold]Validation Report[/bold]")
    console.print("=" * 80)
    
    # Basic statistics
    console.print(f"\n[bold cyan]Summary:[/bold cyan]")
    console.print(f"  Total files in directory: {len(all_files)}")
    console.print(f"  Image files found: {len(image_files)}")
    console.print(f"  Non-image files: {len(non_image_files)}")
    console.print(f"  Total bytes (all files): {total_bytes:,} ({total_bytes / (1024**2):.2f} MB)")
    console.print(f"  Compliant images: {compliant_count}")
    console.print(f"  Non-compliant images: {len(non_compliant_files)}")
    
    # Text file statistics
    total_text_issues = missing_text_files + empty_text_files
    console.print(f"\n[bold cyan]pdfium_text.txt Validation:[/bold cyan]")
    console.print(f"  Missing: {missing_text_files}")
    console.print(f"  Empty: {empty_text_files}")
    console.print(f"  [bold]Total with issues: {total_text_issues}[/bold]")
    if total_text_issues == 0:
        console.print(f"  [green]✓[/green] All images have valid pdfium_text.txt files")
    else:
        console.print(f"  [red]✗[/red] {total_text_issues} images missing or have empty pdfium_text.txt files")
    
    # Image only file statistics
    total_image_only_issues = missing_image_only_files + empty_image_only_files
    console.print(f"\n[bold cyan]pdfium_image_only.txt Validation:[/bold cyan]")
    console.print(f"  Missing: {missing_image_only_files}")
    console.print(f"  Empty: {empty_image_only_files}")
    console.print(f"  [bold]Total with issues: {total_image_only_issues}[/bold]")
    if total_image_only_issues == 0:
        console.print(f"  [green]✓[/green] All images have valid pdfium_image_only.txt files")
    else:
        console.print(f"  [red]✗[/red] {total_image_only_issues} images missing or have empty pdfium_image_only.txt files")
    
    # Expected count check
    if expected_count is not None:
        if len(image_files) == expected_count:
            console.print(f"\n  [green]✓[/green] File count matches expected: {expected_count}")
        else:
            console.print(f"\n  [red]✗[/red] File count mismatch: found {len(image_files)}, expected {expected_count}")
    
    # Format statistics
    if format_stats:
        console.print(f"\n[bold cyan]File Formats:[/bold cyan]")
        for ext, count in sorted(format_stats.items()):
            console.print(f"  {ext}: {count} files")
    
    # Dimension statistics
    if dimension_stats:
        console.print(f"\n[bold cyan]Image Dimensions:[/bold cyan]")
        table = Table(show_header=True)
        table.add_column("Height", style="cyan")
        table.add_column("Width", style="cyan")
        table.add_column("Count", justify="right", style="green")
        
        for (h, w), count in sorted(dimension_stats.items(), key=lambda x: x[1], reverse=True):
            expected_marker = ""
            if expected_height is not None and expected_width is not None:
                if h == expected_height and w == expected_width:
                    expected_marker = " ✓"
            table.add_row(str(h), str(w), f"{count}{expected_marker}")
        
        console.print(table)
    
    # Non-compliant files
    if non_compliant_files:
        console.print(f"\n[bold red]Non-Compliant Files ({len(non_compliant_files)}):[/bold red]")
        table = Table(show_header=True)
        table.add_column("File", style="yellow")
        table.add_column("Reason", style="red")
        table.add_column("Size", justify="right")
        table.add_column("Dimensions", style="cyan")
        table.add_column("Text File", style="magenta")
        
        for item in non_compliant_files:
            table.add_row(
                item["path"],
                item["reason"],
                f"{item['size_bytes']:,}",
                item.get("dimensions", "N/A"),
                item.get("text_file", "N/A"),
            )
        
        console.print(table)
    else:
        console.print(f"\n[bold green]✓ All files are compliant![/bold green]")
    
    # Non-image files list
    if non_image_files:
        console.print(f"\n[bold yellow]Non-Image Files ({len(non_image_files)}):[/bold yellow]")
        for p in non_image_files[:20]:  # Limit to first 20
            console.print(f"  {p.relative_to(input_dir)}")
        if len(non_image_files) > 20:
            console.print(f"  ... and {len(non_image_files) - 20} more")
    
    console.print("\n" + "=" * 80)
    
    # Exit with error code if there are issues
    exit_code = 0
    if expected_count is not None and len(image_files) != expected_count:
        exit_code = 1
    if non_compliant_files:
        exit_code = 1
    
    if exit_code == 0:
        console.print("[bold green]Validation passed![/bold green]")
    else:
        console.print("[bold red]Validation failed - see issues above.[/bold red]")
    
    raise typer.Exit(code=exit_code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

