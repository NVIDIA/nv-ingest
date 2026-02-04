from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm

from retriever.model.local.nemotron_graphic_elements_v1 import NemotronGraphicElementsV1

from ._io import (
    bbox_region_to_page,
    crop_tensor_normalized_xyxy,
    iter_detection_dicts,
    iter_images,
    load_image_rgb_chw_u8,
    read_json,
    to_bbox_list,
    to_scalar_float,
    to_scalar_int,
    write_json,
)


install(show_locals=False)
console = Console()
app = typer.Typer(help="Stage 3: crop graphic regions from stage2 and run graphic_elements_v1; save JSON per image.")

DEFAULT_INPUT_DIR = Path("./data/pages")


def _stage2_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".page_elements_v3.json")


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".graphic_elements_v1.json")


def _is_graphic_label(label: int) -> bool:
    # # In local/simple.py: label 0 is table; labels in [1,2,3] treated as "graphic".
    # return int(label) in (1, 2, 3)
    # 0 -> table
    # 1 -> chart
    # 2 -> text
    # 3 -> infographic
    # 4 -> header_footer
    return int(label) in (1, 3)


def _count_graphic_region_detections_from_stage2_json(s2: Any) -> int:
    """
    Stage2 outputs include:
      - detections[].label_name (preferred)
      - detections[].label (fallback)

    For stage3, we count regions that will be processed here: label_name in {"chart", "infographic"}
    (case-insensitive). If label_name is missing, we fall back to label in {1, 3}.
    """
    if not isinstance(s2, dict):
        return 0
    dets = s2.get("detections") or []
    if not isinstance(dets, list):
        return 0
    n = 0
    for d in dets:
        if not isinstance(d, dict):
            continue
        name = d.get("label_name")
        if isinstance(name, str):
            nm = name.strip().lower()
            if nm in ("chart", "infographic"):
                n += 1
                continue
        # Fallback for older stage2 payloads (or if label_name is absent)
        if name is None:
            try:
                if int(d.get("label", -1)) in (1, 3):
                    n += 1
            except Exception:
                pass
    return int(n)


def _precount_total_graphic_region_detections(images: Sequence[Path]) -> Tuple[int, Dict[Path, int]]:
    """
    Pre-scan stage2 sidecars to compute:
      - total number of chart/infographic detections across images
      - per-image chart/infographic detection counts (used to update progress without re-reading stage2 on skips)
    """
    counts: Dict[Path, int] = {}
    total = 0
    for img_path in images:
        stage2_path = _stage2_json_for_image(img_path)
        if not stage2_path.exists():
            counts[img_path] = 0
            continue
        try:
            s2 = read_json(stage2_path)
        except Exception:
            counts[img_path] = 0
            continue
        n = _count_graphic_region_detections_from_stage2_json(s2)
        counts[img_path] = int(n)
        total += int(n)
    return int(total), counts


def _chunked(seq: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def _invoke_batched_graphic_elements(
    model: NemotronGraphicElementsV1,
    batch_tensor: torch.Tensor,
    orig_shapes: Sequence[Tuple[int, int]],
) -> List[Any]:
    """
    Best-effort batched inference for graphic_elements_v1.

    Tries calling the underlying model with (BCHW, list[shape]) then (BCHW, shape).
    Normalizes output into a per-image list (len == batch size), or raises to allow fallback.
    """
    m = getattr(model, "_model", None)
    if m is None:
        raise RuntimeError("Model does not expose _model for batched call.")

    try:
        raw = m(batch_tensor, list(orig_shapes))
    except Exception:
        raw = m(batch_tensor, orig_shapes[0])

    raw0 = raw[0] if isinstance(raw, (tuple, list)) and len(raw) > 0 else raw
    if isinstance(raw0, list) and len(raw0) == int(batch_tensor.shape[0]):
        return raw0
    raise RuntimeError("Batched model output was not per-image list; falling back to per-region inference.")


def _ensure_bchw(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(f"Expected CHW or BCHW, got shape {tuple(t.shape)}")


@app.command()
def validate_input(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    expected_table: Optional[int] = typer.Option(None, "--expected-table", help="Expected number of table detections."),
    expected_chart: Optional[int] = typer.Option(None, "--expected-chart", help="Expected number of chart detections."),
    expected_text: Optional[int] = typer.Option(None, "--expected-text", help="Expected number of text detections."),
    expected_infographic: Optional[int] = typer.Option(
        None, "--expected-infographic", help="Expected number of infographic detections."
    ),
    expected_title: Optional[int] = typer.Option(None, "--expected-title", help="Expected number of title detections."),
):
    """
    Validate all .page_elements_v3.json files in input_dir: count detections by label type.

    This command scans the input directory for .page_elements_v3.json files and generates a validation report showing:
    - Total number of detections for each label type (table, chart, text, infographic, title)
    - Comparison against expected counts (if provided)
    - List of files with detections
    """
    console.print(f"[bold cyan]Validating page_elements_v3 inputs[/bold cyan] in {input_dir}")

    # Initialize counters for each label type
    label_counts: Dict[str, int] = {
        "table": 0,
        "chart": 0,
        "text": 0,
        "infographic": 0,
        "title": 0,
        "header_footer": 0,
        "unknown": 0,
    }

    # Track files with detections
    files_with_detections: List[Tuple[Path, Dict[str, int]]] = []

    # Find all .page_elements_v3.json files
    images = iter_images(input_dir)
    json_files: List[Path] = []
    for img_path in images:
        json_path = _stage2_json_for_image(img_path)
        if json_path.exists():
            json_files.append(json_path)

    console.print(f"Found {len(json_files)} .page_elements_v3.json files")

    if len(json_files) == 0:
        console.print("[yellow]Warning: No .page_elements_v3.json files found in input directory.[/yellow]")
        raise typer.Exit(code=1)

    # Process each JSON file
    pbar = tqdm(json_files, desc="Scanning JSON files", unit="file")
    for json_path in pbar:
        file_name = json_path.name[:50] + "..." if len(json_path.name) > 50 else json_path.name
        pbar.set_postfix({"current": file_name})

        try:
            data = read_json(json_path)

            if not isinstance(data, dict):
                console.print(f"[yellow]Warning: {json_path.name} does not contain a valid JSON object[/yellow]")
                continue

            detections = data.get("detections", [])
            if not isinstance(detections, list):
                console.print(f"[yellow]Warning: {json_path.name} detections field is not a list[/yellow]")
                continue

            # Count detections by label type for this file
            file_label_counts: Dict[str, int] = {
                "table": 0,
                "chart": 0,
                "text": 0,
                "infographic": 0,
                "title": 0,
                "header_footer": 0,
                "unknown": 0,
            }

            for det in detections:
                if not isinstance(det, dict):
                    continue

                # Try to get label_name first (preferred), then fall back to label
                label_name = det.get("label_name")
                if isinstance(label_name, str):
                    label_name_lower = label_name.strip().lower()
                    if label_name_lower in file_label_counts:
                        file_label_counts[label_name_lower] += 1
                        label_counts[label_name_lower] += 1
                    else:
                        file_label_counts["unknown"] += 1
                        label_counts["unknown"] += 1
                else:
                    # Fall back to label integer
                    label = det.get("label")
                    if label is not None:
                        try:
                            label_int = int(label)
                            # Map using PAGE_ELEMENT_LABELS from _io.py
                            # 0: table, 1: chart, 2: title, 3: infographic, 4: text, 5: header_footer
                            label_map = {
                                0: "table",
                                1: "chart",
                                2: "title",
                                3: "infographic",
                                4: "text",
                                5: "header_footer",
                            }
                            if label_int in label_map:
                                mapped_name = label_map[label_int]
                                file_label_counts[mapped_name] += 1
                                label_counts[mapped_name] += 1
                            else:
                                file_label_counts["unknown"] += 1
                                label_counts["unknown"] += 1
                        except (ValueError, TypeError):
                            file_label_counts["unknown"] += 1
                            label_counts["unknown"] += 1

            # Only track files that have at least one detection
            if sum(file_label_counts.values()) > 0:
                files_with_detections.append((json_path, file_label_counts))

        except Exception as e:
            console.print(f"[red]Error processing {json_path.name}: {str(e)}[/red]")
            continue

    pbar.close()

    # Print summary report
    console.print("\n" + "=" * 80)
    console.print("[bold]Validation Report: Page Elements V3[/bold]")
    console.print("=" * 80)

    # Detection counts summary
    console.print(f"\n[bold cyan]Detection Counts by Label Type:[/bold cyan]")
    from rich.table import Table

    table = Table(show_header=True)
    table.add_column("Label Type", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Expected", justify="right", style="yellow")
    table.add_column("Status", justify="center")

    expected_map = {
        "table": expected_table,
        "chart": expected_chart,
        "text": expected_text,
        "infographic": expected_infographic,
        "title": expected_title,
    }

    validation_passed = True
    for label_type in ["table", "chart", "text", "infographic", "title", "header_footer", "unknown"]:
        count = label_counts[label_type]
        expected = expected_map.get(label_type)

        status = ""
        if expected is not None:
            if count == expected:
                status = "[green]✓[/green]"
            else:
                status = f"[red]✗ (diff: {count - expected:+d})[/red]"
                validation_passed = False

        table.add_row(
            label_type,
            str(count),
            str(expected) if expected is not None else "N/A",
            status,
        )

    console.print(table)

    # Total detections
    total_detections = sum(label_counts.values())
    console.print(f"\n[bold]Total detections across all files: {total_detections}[/bold]")
    console.print(f"[bold]Files with detections: {len(files_with_detections)} / {len(json_files)}[/bold]")

    # Show sample of files with most detections
    if files_with_detections:
        console.print(f"\n[bold cyan]Top 10 Files by Detection Count:[/bold cyan]")
        files_with_detections.sort(key=lambda x: sum(x[1].values()), reverse=True)

        detail_table = Table(show_header=True)
        detail_table.add_column("File", style="cyan", max_width=40)
        detail_table.add_column("Table", justify="right")
        detail_table.add_column("Chart", justify="right")
        detail_table.add_column("Text", justify="right")
        detail_table.add_column("Infographic", justify="right")
        detail_table.add_column("Title", justify="right")
        detail_table.add_column("Total", justify="right", style="bold green")

        for json_path, file_counts in files_with_detections[:10]:
            detail_table.add_row(
                json_path.name,
                str(file_counts["table"]),
                str(file_counts["chart"]),
                str(file_counts["text"]),
                str(file_counts["infographic"]),
                str(file_counts["title"]),
                str(sum(file_counts.values())),
            )

        console.print(detail_table)

    console.print("\n" + "=" * 80)

    # Exit with appropriate code
    exit_code = 0
    if not validation_passed:
        console.print("[bold red]Validation failed - detection counts do not match expected values.[/bold red]")
        exit_code = 1
    else:
        console.print("[bold green]Validation passed![/bold green]")

    raise typer.Exit(code=exit_code)


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for model + tensors."),
    region_batch_size: int = typer.Option(
        32, "--region-batch-size", min=1, help="Best-effort batch size for region crops (within each page)."
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Optional graphic elements NIM endpoint URL. If set, runs remotely and local weights are not loaded.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
):
    dev = torch.device(device)
    model = NemotronGraphicElementsV1(endpoint=endpoint, remote_batch_size=region_batch_size)

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    total_region_dets, region_dets_by_image = _precount_total_graphic_region_detections(images)
    console.print(
        f"[bold cyan]Stage3[/bold cyan] images={len(images)} "
        f"chart_infographic_detections_total={total_region_dets} input_dir={input_dir} device={dev}"
    )

    processed = 0
    skipped = 0
    missing_stage2 = 0
    bad_stage2 = 0
    pbar = tqdm(total=total_region_dets, desc="Stage3 progress", unit="det")
    for img_i, img_path in enumerate(images, start=1):
        pbar.set_postfix(img=f"{img_i}/{len(images)}", file=img_path.name)
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            pbar.update(int(region_dets_by_image.get(img_path, 0)))
            continue

        stage2_path = _stage2_json_for_image(img_path)
        if not stage2_path.exists():
            missing_stage2 += 1
            continue

        try:
            s2 = read_json(stage2_path)
        except Exception:
            bad_stage2 += 1
            continue
        dets: Sequence[Dict[str, Any]] = s2.get("detections") or []

        t, (h, w) = load_image_rgb_chw_u8(img_path, dev)

        # Collect candidate graphic regions first so we can micro-batch them.
        region_jobs: List[Dict[str, Any]] = []
        for region_idx, d in enumerate(dets):
            lab = int(d.get("label", -1))
            if not _is_graphic_label(lab):
                continue
            region_bbox = d.get("bbox_xyxy_norm")
            if not region_bbox or len(region_bbox) != 4:
                continue
            region_jobs.append(
                {
                    "region_index": int(region_idx),
                    "region_bbox": list(region_bbox),
                    "page_element": {
                        "bbox_xyxy_norm": list(region_bbox),
                        "label": int(lab),
                        "score": d.get("score"),
                    },
                }
            )

        regions_out: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for job_batch in _chunked(region_jobs, region_batch_size):
                inputs: List[torch.Tensor] = []
                shapes: List[Tuple[int, int]] = []
                for job in job_batch:
                    region_bbox = job["region_bbox"]
                    crop = crop_tensor_normalized_xyxy(t, region_bbox)
                    shapes.append((int(crop.shape[1]), int(crop.shape[2])))  # (H, W)
                    inputs.append(model.preprocess(crop))

                per_region_preds: Optional[List[Any]] = None
                try:
                    batch_tensor = torch.cat([_ensure_bchw(x) for x in inputs], dim=0)
                    per_region_preds = _invoke_batched_graphic_elements(model, batch_tensor, shapes)
                except Exception:
                    per_region_preds = None

                for i, job in enumerate(job_batch):
                    region_bbox = job["region_bbox"]
                    crop_shape = shapes[i]
                    preds = per_region_preds[i] if per_region_preds is not None else model.invoke(inputs[i], crop_shape)

                    det_list: List[Dict[str, Any]] = []
                    for det in iter_detection_dicts(preds):
                        det_boxes = det.get("boxes", None)
                        det_labels = det.get("labels", None)
                        det_scores = det.get("scores", None)
                        if det_boxes is None:
                            det_boxes = []
                        if det_labels is None:
                            det_labels = []
                        if det_scores is None:
                            det_scores = []
                        n = min(len(det_boxes), len(det_labels)) if hasattr(det_boxes, "__len__") else 0
                        for j in range(int(n)):
                            bbox_in_region = to_bbox_list(det_boxes[j])
                            score = det_scores[j] if j < len(det_scores) else None
                            det_list.append(
                                {
                                    "bbox_xyxy_norm_in_region": bbox_in_region,
                                    "bbox_xyxy_norm_in_page": bbox_region_to_page(
                                        region_bbox_xyxy_norm_in_page=region_bbox,
                                        det_bbox_xyxy_norm_in_region=bbox_in_region,
                                    ),
                                    "label": to_scalar_int(det_labels[j]),
                                    "score": to_scalar_float(score),
                                }
                            )

                    regions_out.append(
                        {
                            "region_index": job["region_index"],
                            "page_element": job["page_element"],
                            "detections": det_list,
                            "raw_preds_present": preds is not None,
                        }
                    )
        dt = time.perf_counter() - t0

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "stage": 3,
            "model": "graphic_elements_v1",
            "image": {"path": str(img_path), "height": int(h), "width": int(w)},
            "stage2_json": str(stage2_path),
            "regions": regions_out,
            "timing": {"seconds": float(dt)},
        }
        write_json(out_path, payload)
        processed += 1
        pbar.update(int(region_dets_by_image.get(img_path, 0)))

    pbar.close()
    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_stage2={missing_stage2} bad_stage2={bad_stage2} wrote_json_suffix=.graphic_elements_v1.json"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
