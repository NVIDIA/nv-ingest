from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.traceback import install
import torch
import torch.nn.functional as F
import typer
from tqdm import tqdm

# from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1
from nemotron_ocr.inference.pipeline import NemotronOCR

from ._io import crop_tensor_normalized_xyxy, load_image_rgb_chw_u8, read_json, write_json


install(show_locals=False)
console = Console()
app = typer.Typer(help="Stage 5: run nemotron_ocr_v1 over stage2 table/chart/infographic detections; save JSON per image.")

DEFAULT_INPUT_DIR = Path("./data/pages")


def _resize_pad_tensor(img: torch.Tensor, target_hw: Tuple[int, int] = (1024, 1024), pad_value: float = 114.0) -> torch.Tensor:
    """
    Resize+pad CHW image tensor to fixed size (preserve aspect ratio).
    Returns uint8 CHW tensor.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    C, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
    th, tw = int(target_hw[0]), int(target_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image shape: {tuple(img.shape)}")

    x = img.float()
    scale = min(th / H, tw / W)
    nh = max(1, int(H * scale))
    nw = max(1, int(W * scale))
    x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    x = torch.clamp(x, 0, 255)
    pad_b = th - nh
    pad_r = tw - nw
    x = F.pad(x, (0, pad_r, 0, pad_b), value=float(pad_value))
    return x.to(dtype=torch.uint8)


STAGE2_SUFFIX = ".page_elements_v3.json"
OUT_SUFFIX = ".nemotron_ocr_v1.json"


class _Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_s(self) -> float:
        return float(time.perf_counter() - self._t0)


class _TimerBank:
    def __init__(self) -> None:
        self._acc: Dict[str, float] = {}

    def add(self, key: str, dt_s: float) -> None:
        self._acc[key] = float(self._acc.get(key, 0.0) + float(dt_s))

    def timed(self, key: str):
        bank = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner._t0 = time.perf_counter()
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                bank.add(key, time.perf_counter() - self_inner._t0)
                return False

        return _Ctx()

    def as_dict(self) -> Dict[str, float]:
        return dict(sorted(self._acc.items(), key=lambda kv: kv[0]))


def _resolve_image_path(input_dir: Path, json_path: Path, blob: Any, suffix: str) -> Path:
    # 1) Prefer the embedded path if it exists.
    if isinstance(blob, dict):
        img = blob.get("image")
        if isinstance(img, dict):
            p = img.get("path")
            if isinstance(p, str) and p.strip():
                cand = Path(p)
                if cand.exists():
                    return cand

    # 2) Infer from filename convention: <image_name> + suffix
    name = json_path.name
    if name.endswith(suffix):
        img_name = name[: -len(suffix)]
        cand = input_dir / img_name
        return cand

    # 3) Last resort: put it next to the JSON and strip suffix.
    return json_path.with_name(name[: -len(suffix)])


def _iter_stage_jsons(input_dir: Path, suffix: str) -> List[Path]:
    out: List[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.name.endswith(suffix):
            out.append(p)
    return out


_STAGE2_LABELS_INCLUDE = {0, 1, 3}  # table, chart, infographic
_STAGE2_LABEL_NAMES_INCLUDE = {"table", "chart", "infographic"}


def _is_stage2_target_detection(det: Dict[str, Any]) -> bool:
    try:
        lab = det.get("label", None)
        if lab is not None:
            # tolerate float labels and string numerals
            if int(lab) in _STAGE2_LABELS_INCLUDE:
                return True
    except Exception:
        pass

    nm = det.get("label_name", None)
    if isinstance(nm, str) and nm.strip().lower() in _STAGE2_LABEL_NAMES_INCLUDE:
        return True
    return False


def _iter_target_detections_from_stage2_json(blob: Any) -> Iterable[Tuple[Dict[str, Any], List[float]]]:
    """
    Yields (detection_dict, bbox_xyxy_norm_in_page).
    Expects stage2 schema:
      - detections[].bbox_xyxy_norm
      - detections[].label
      - detections[].label_name
    """
    if not isinstance(blob, dict):
        return []
    dets = blob.get("detections") or []
    if not isinstance(dets, list):
        return []
    for det in dets:
        if not isinstance(det, dict):
            continue
        if not _is_stage2_target_detection(det):
            continue
        bbox = det.get("bbox_xyxy_norm")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            bb = [float(x) for x in bbox]
        except Exception:
            continue
        yield (det, bb)


def _extract_text_best_effort(raw: Any) -> str:
    """
    Nemotron OCR wrapper shapes vary:
      - remote: {"text": "...", "raw": ...} per image
      - local: often list[dict] with line-level fields
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for k in ("text", "output_text", "generated_text", "ocr_text"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # remote wrapper uses {"text": ..., "raw": ...}
        if "raw" in raw:
            return _extract_text_best_effort(raw.get("raw"))
        return str(raw).strip()
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            t = _extract_text_best_effort(item)
            if t:
                parts.append(t)
        return " ".join(parts).strip()
    return str(raw).strip()


def _out_path(output_dir: Path, img_path: Path) -> Path:
    return output_dir / (img_path.name + OUT_SUFFIX)


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", file_okay=False, help="Directory to write stage5 JSON outputs."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for tensors/model (single-device only)."),
    batch_size: int = typer.Option(16, "--batch-size", min=1, help="Target number of crops per OCR batch."),
    ocr_model_dir: Path = typer.Option(
        # Path("/raid/jdyer/slimgest/models/nemotron-ocr-v1/checkpoints"),
        Path("/raid/slimgest/models/models--nvidia--nemotron-ocr-v1/snapshots/90015d3b851ba898ca842f18e948690af49c2427/checkpoints"),
        "--ocr-model-dir",
        help="Local nemotron-ocr-v1 checkpoints directory (ignored if --ocr-endpoint is set).",
    ),
    ocr_endpoint: Optional[str] = typer.Option(
        None,
        "--ocr-endpoint",
        help="Optional OCR NIM endpoint URL. If set, OCR runs remotely and local weights are not loaded.",
    ),
    remote_batch_size: int = typer.Option(32, "--remote-batch-size", help="Remote OCR internal chunk size when using --ocr-endpoint."),
    resize_to_1024: bool = typer.Option(True, "--resize-to-1024/--no-resize-to-1024", help="Resize+pad crops to 1024x1024 before OCR."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of images processed."),
):
    dev = torch.device(device)
    out_dir = Path(output_dir) if output_dir is not None else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timers = _TimerBank()
    sw_total = _Stopwatch()

    with timers.timed("01_discover_jsons"):
        stage2_jsons = _iter_stage_jsons(input_dir, STAGE2_SUFFIX)

    # Build per-image task lists from stage2 JSON sidecars.
    tasks_by_image: Dict[Path, List[Dict[str, Any]]] = {}
    prereq_by_image: Dict[Path, Dict[str, Optional[Path]]] = {}
    bad_json = 0
    missing_image = 0
    n_stage2_target_dets = 0

    def _add_tasks_from_stage2_json(json_path: Path) -> None:
        nonlocal bad_json, missing_image, n_stage2_target_dets
        with timers.timed("02_read_json"):
            try:
                blob = read_json(json_path)
            except Exception:
                bad_json += 1
                return
        img_path = _resolve_image_path(input_dir, json_path, blob, STAGE2_SUFFIX)
        if not img_path.exists():
            missing_image += 1
            return

        for det, bbox in _iter_target_detections_from_stage2_json(blob):
            # Keep a stable page-element-like record (stage2 outputs are already page-level)
            page_el = {
                "bbox_xyxy_norm": det.get("bbox_xyxy_norm"),
                "label": det.get("label"),
                "label_name": det.get("label_name"),
                "score": det.get("score"),
            }
            rec = {
                "source_model": "page_elements_v3",
                "bbox_xyxy_norm_in_page": bbox,
                "page_element": page_el,
                "detection": det,
                "source_json": str(json_path),
            }
            tasks_by_image.setdefault(img_path, []).append(rec)
            prereq_by_image.setdefault(img_path, {"stage2_json": None})
            prereq_by_image[img_path]["stage2_json"] = json_path
            n_stage2_target_dets += 1

    with timers.timed("03_parse_stage2"):
        for p in stage2_jsons:
            _add_tasks_from_stage2_json(p)

    # Include all images from stage2 JSONs, even those with no target detections
    all_images_from_jsons = set()
    for json_path in stage2_jsons:
        try:
            blob = read_json(json_path)
            img_path = _resolve_image_path(input_dir, json_path, blob, STAGE2_SUFFIX)
            if img_path.exists():
                all_images_from_jsons.add(img_path)
                # Ensure image is in prereq_by_image even if it has no tasks
                if img_path not in prereq_by_image:
                    prereq_by_image[img_path] = {"stage2_json": json_path}
        except Exception:
            pass

    # Combine images with tasks and images without tasks
    images = sorted(all_images_from_jsons.union(tasks_by_image.keys()))
    if limit is not None:
        images = images[: int(limit)]

    total_crops = int(sum(len(tasks_by_image.get(p, [])) for p in images))
    total_batches = int((total_crops + int(batch_size) - 1) // int(batch_size)) if total_crops > 0 else 0

    console.print(
        f"[bold cyan]Stage5[/bold cyan] input_dir={input_dir} output_dir={out_dir} device={dev} "
        f"images_with_detections={len(images)} total_detections={total_crops} "
        f"total_batches(ceil(dets/batch_size))={total_batches} batch_size={int(batch_size)} "
        f"stage2_jsons={len(stage2_jsons)} stage2_target_dets={n_stage2_target_dets} bad_json={bad_json} missing_image={missing_image} "
        f"ocr={'remote' if ocr_endpoint else 'local'} resize_to_1024={resize_to_1024}"
    )

    if total_crops == 0:
        console.print("[yellow]No table/chart/infographic detections found in stage2 JSONs.[/yellow]")
        # Don't return early - we still want to create empty output files for all images
        # Skip model initialization since we won't need OCR
        ocr = None
    else:
        with timers.timed("05_init_model"):
            ocr = NemotronOCR(
                model_dir=str(ocr_model_dir),
                # endpoint=str(ocr_endpoint).strip() if ocr_endpoint else None,
                # remote_batch_size=int(remote_batch_size),
            )
    processed_images = 0
    skipped_images = 0
    processed_crops = 0
    processed_batches = 0
    failed_images = 0

    p_imgs = tqdm(total=len(images), desc="Stage5 images", unit="img")
    p_batches = tqdm(total=total_batches, desc="Stage5 OCR batches", unit="batch")

    sw_ocr = _Stopwatch()

    for img_i, img_path in enumerate(images, start=1):
        out_path = _out_path(out_dir, img_path)
        if out_path.exists() and not overwrite:
            skipped_images += 1
            p_imgs.update(1)
            p_imgs.set_postfix(img=f"{img_i}/{len(images)}", skipped=skipped_images)
            continue

        tasks = tasks_by_image.get(img_path, [])
        if not tasks:
            # Write an empty output file even when there are no detections
            # so downstream scripts have a file to open
            try:
                with timers.timed("06_load_image"):
                    page_tensor, (h, w) = load_image_rgb_chw_u8(img_path, dev)
                
                payload: Dict[str, Any] = {
                    "schema_version": 1,
                    "stage": 5,
                    "model": "nemotron_ocr_v1",
                    "image": {"path": str(img_path), "height": int(h), "width": int(w)},
                    "stage2_json": str((prereq_by_image.get(img_path) or {}).get("stage2_json") or ""),
                    "regions": [],  # Empty regions list
                    "run": {
                        "device": str(dev),
                        "batch_size": int(batch_size),
                        "resize_to_1024": bool(resize_to_1024),
                        "ocr_endpoint": str(ocr_endpoint) if ocr_endpoint else None,
                        "remote_batch_size": int(remote_batch_size),
                    },
                }
                with timers.timed("11_write_json"):
                    write_json(out_path, payload)
                
                processed_images += 1
            except Exception as e:
                failed_images += 1
                console.print(f"[red]Stage5 failed[/red] file={img_path.name} err={type(e).__name__}: {e}")
            finally:
                p_imgs.update(1)
            continue

        p_imgs.set_postfix(img=f"{img_i}/{len(images)}", file=img_path.name, dets=len(tasks))

        # This should not happen if total_crops > 0, but add a safeguard
        if ocr is None:
            console.print(f"[red]Unexpected: tasks present but OCR model not initialized[/red] file={img_path.name}")
            failed_images += 1
            p_imgs.update(1)
            continue

        try:
            with timers.timed("06_load_image"):
                page_tensor, (h, w) = load_image_rgb_chw_u8(img_path, dev)

            # Build crop tensors in task order.
            crops: List[torch.Tensor] = []
            with torch.inference_mode():
                for rec in tasks:
                    bbox = rec["bbox_xyxy_norm_in_page"]
                    with timers.timed("07_crop"):
                        crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
                    if resize_to_1024:
                        with timers.timed("08_resize_pad"):
                            crop = _resize_pad_tensor(crop, target_hw=(1024, 1024))
                    crops.append(crop)

                # OCR batches
                ocr_raw_outs: List[Any] = []
                for j in range(0, len(crops), int(batch_size)):
                    batch = crops[j : j + int(batch_size)]
                    if not batch:
                        continue
                    t_batch0 = time.perf_counter()
                    with timers.timed("09_model_invoke"):
                        if ocr_endpoint:
                            b = torch.stack([t if t.ndim == 3 else t.squeeze(0) for t in batch], dim=0)
                            ocr_raw_outs.extend(list(ocr._process_tensor(b, "paragraph")))
                        else:
                            # Local pipeline is effectively per-image; keep "batch" boundaries
                            # for reporting/throughput visibility.
                            for t in batch:
                                ocr_raw_outs.append(ocr._process_tensor(t, merge_level="paragraph"))
                    batch_dt = max(1e-9, time.perf_counter() - t_batch0)

                    processed_batches += 1
                    processed_crops += int(len(batch))
                    p_batches.update(1)

                    # Performance visibility (focus: pages/sec and batch cadence)
                    elapsed = max(1e-9, sw_ocr.elapsed_s())
                    pages_per_s = processed_crops / elapsed
                    batch_pages_per_s = len(batch) / batch_dt
                    p_batches.set_postfix(
                        pages_s=f"{pages_per_s:.2f}",
                        batch_pages_s=f"{batch_pages_per_s:.2f}",
                        crops=f"{processed_crops}/{total_crops}",
                        bs=len(batch),
                    )

            if len(ocr_raw_outs) != len(tasks):
                raise RuntimeError(f"OCR output count mismatch: got {len(ocr_raw_outs)} outputs for {len(tasks)} crops.")

            with timers.timed("10_postprocess"):
                regions_out: List[Dict[str, Any]] = []
                for rec, raw in zip(tasks, ocr_raw_outs):
                    regions_out.append(
                        {
                            "source_model": rec.get("source_model"),
                            "bbox_xyxy_norm_in_page": rec.get("bbox_xyxy_norm_in_page"),
                            "page_element": rec.get("page_element"),
                            "detection": rec.get("detection"),
                            "source_json": rec.get("source_json"),
                            "ocr_raw": raw,
                            "ocr_text": _extract_text_best_effort(raw),
                        }
                    )

            payload: Dict[str, Any] = {
                "schema_version": 1,
                "stage": 5,
                "model": "nemotron_ocr_v1",
                "image": {"path": str(img_path), "height": int(h), "width": int(w)},
                "stage2_json": str((prereq_by_image.get(img_path) or {}).get("stage2_json") or ""),
                "regions": regions_out,
                "run": {
                    "device": str(dev),
                    "batch_size": int(batch_size),
                    "resize_to_1024": bool(resize_to_1024),
                    "ocr_endpoint": str(ocr_endpoint) if ocr_endpoint else None,
                    "remote_batch_size": int(remote_batch_size),
                },
            }
            with timers.timed("11_write_json"):
                write_json(out_path, payload)

            processed_images += 1
        except Exception as e:
            failed_images += 1
            console.print(f"[red]Stage5 failed[/red] file={img_path.name} err={type(e).__name__}: {e}")
        finally:
            p_imgs.update(1)

    p_imgs.close()
    p_batches.close()

    wall_s = sw_total.elapsed_s()
    pages_s = (processed_crops / max(1e-9, sw_ocr.elapsed_s())) if processed_crops else 0.0

    report = {
        "schema_version": 1,
        "stage": 5,
        "model": "nemotron_ocr_v1",
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "device": str(dev),
        "batch_size": int(batch_size),
        "resize_to_1024": bool(resize_to_1024),
        "ocr": {
            "mode": "remote" if ocr_endpoint else "local",
            "endpoint": str(ocr_endpoint) if ocr_endpoint else None,
            "model_dir": str(ocr_model_dir),
            "remote_batch_size": int(remote_batch_size),
        },
        "counts": {
            "stage2_jsons": int(len(stage2_jsons)),
            "stage2_target_detections": int(n_stage2_target_dets),
            "images_with_detections": int(len(images)),
            "processed_images": int(processed_images),
            "skipped_images": int(skipped_images),
            "failed_images": int(failed_images),
            "total_detections": int(total_crops),
            "processed_crops": int(processed_crops),
            "total_batches": int(total_batches),
            "processed_batches": int(processed_batches),
            "bad_json": int(bad_json),
            "missing_image": int(missing_image),
        },
        "throughput": {
            "pages_per_second": float(pages_s),
        },
        "timing_s": timers.as_dict(),
        "wall_s": float(wall_s),
    }

    with timers.timed("12_write_report"):
        report_path = out_dir / "stage5_nemotron_ocr_v1.report.json"
        write_json(report_path, report)

    console.print(
        f"[green]Done[/green] processed_images={processed_images} skipped_images={skipped_images} failed_images={failed_images} "
        f"processed_crops={processed_crops}/{total_crops} processed_batches={processed_batches}/{total_batches} "
        f"pages_per_second={pages_s:.2f} report={report_path}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

