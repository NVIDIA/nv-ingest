from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm

from slimgest.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1

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
app = typer.Typer(help="Stage 4: crop table regions from stage2 and run table_structure_v1; save JSON per image.")

DEFAULT_INPUT_DIR = Path("./data/pages")


def _stage2_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".page_elements_v3.json")


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".table_structure_v1.json")


def _is_table_label(label: int) -> bool:
    # In local/simple.py: label 0 treated as "table".
    return int(label) == 0


def _chunked(seq: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def _ensure_bchw(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 4:
        return t
    raise ValueError(f"Expected CHW or BCHW, got shape {tuple(t.shape)}")


def _invoke_batched_table_structure(
    model: NemotronTableStructureV1,
    batch_tensor: torch.Tensor,
    orig_shapes: Sequence[Tuple[int, int]],
) -> List[Any]:
    """
    Best-effort batched inference for table_structure_v1.
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
        help="Optional table structure NIM endpoint URL. If set, runs remotely and local weights are not loaded.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
):
    dev = torch.device(device)
    model = NemotronTableStructureV1(endpoint=endpoint, remote_batch_size=region_batch_size)

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(f"[bold cyan]Stage4[/bold cyan] images={len(images)} input_dir={input_dir} device={dev}")

    processed = 0
    skipped = 0
    missing_stage2 = 0
    bad_stage2 = 0
    for img_path in tqdm(images, desc="Stage4 images", unit="img"):
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
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
        dets: Sequence[Dict[str, Any]] = (s2.get("detections") or [])

        t, (h, w) = load_image_rgb_chw_u8(img_path, dev)

        # Collect candidate table regions first so we can micro-batch them.
        region_jobs: List[Dict[str, Any]] = []
        for region_idx, d in enumerate(dets):
            lab = int(d.get("label", -1))
            if not _is_table_label(lab):
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
                    crop_shape = (int(crop.shape[1]), int(crop.shape[2]))  # (H, W)
                    shapes.append(crop_shape)
                    inputs.append(model.preprocess(crop, crop_shape))

                per_region_preds: Optional[List[Any]] = None
                try:
                    batch_tensor = torch.cat([_ensure_bchw(x) for x in inputs], dim=0)
                    per_region_preds = _invoke_batched_table_structure(model, batch_tensor, shapes)
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
            "stage": 4,
            "model": "table_structure_v1",
            "image": {"path": str(img_path), "height": int(h), "width": int(w)},
            "stage2_json": str(stage2_path),
            "regions": regions_out,
            "timing": {"seconds": float(dt)},
        }
        write_json(out_path, payload)
        processed += 1

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_stage2={missing_stage2} bad_stage2={bad_stage2} wrote_json_suffix=.table_structure_v1.json"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

