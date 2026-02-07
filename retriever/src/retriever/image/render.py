from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from tqdm import tqdm

console = Console()
app = typer.Typer(help="Render/visualize various model outputs on images.")


def _stage2_json_path_for_image(img_path: Path) -> Path:
    # Matches stage2_page_elements_v3.py naming.
    return img_path.with_name(img_path.name + ".page_elements_v3.json")


def _out_path_for_image(img_path: Path) -> Path:
    # Matches requested suffix: "page_element_detections.png"
    return img_path.with_name(img_path.name + ".page_element_detections.png")


def _clamp_bbox_xyxy_px(bbox: Sequence[float], *, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    x1_i = int(round(x1))
    y1_i = int(round(y1))
    x2_i = int(round(x2))
    y2_i = int(round(y2))

    x1_i = max(0, min(x1_i, w - 1))
    y1_i = max(0, min(y1_i, h - 1))
    x2_i = max(x1_i + 1, min(x2_i, w))
    y2_i = max(y1_i + 1, min(y2_i, h))
    return x1_i, y1_i, x2_i, y2_i


def _bbox_xyxy_norm_to_px(bbox_xyxy_norm: Sequence[float], *, w: int, h: int) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    x1 = x1n * w
    y1 = y1n * h
    x2 = x2n * w
    y2 = y2n * h
    return _clamp_bbox_xyxy_px((x1, y1, x2, y2), w=w, h=h)


def _label_color(label_name: str) -> Tuple[int, int, int]:
    # High-contrast palette.
    palette: Dict[str, Tuple[int, int, int]] = {
        "table": (0, 170, 255),
        "chart": (255, 170, 0),
        "title": (0, 220, 120),
        "infographic": (200, 100, 255),
        "text": (255, 80, 80),
        "header_footer": (255, 255, 0),
    }
    return palette.get(label_name, (255, 0, 0))


@dataclass(frozen=True)
class PageElementDetection:
    bbox_xyxy_norm: Tuple[float, float, float, float]
    label: Optional[int]
    label_name: str
    score: Optional[float]


def _iter_page_element_detections(payload: Any) -> Iterable[PageElementDetection]:
    if not isinstance(payload, dict):
        return []
    dets = payload.get("detections", [])
    if not isinstance(dets, list):
        return []

    out: List[PageElementDetection] = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox_xyxy_norm")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            bbox_f = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except Exception:
            continue
        label = d.get("label")
        try:
            label_i = int(label) if label is not None else None
        except Exception:
            label_i = None
        label_name = str(d.get("label_name") or (f"label_{label_i}" if label_i is not None else "unknown"))
        score = d.get("score")
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        out.append(
            PageElementDetection(
                bbox_xyxy_norm=bbox_f,
                label=label_i,
                label_name=label_name,
                score=score_f,
            )
        )
    return out


def render_page_element_detections_for_image(
    *,
    image_path: Path,
    stage2_json_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    min_score: float = 0.0,
    line_width: int = 3,
    draw_labels: bool = True,
) -> Optional[Path]:
    """
    Read stage2 page_elements_v3 JSON and render bbox overlays onto the original image.

    Output file naming follows:
      <image>.<ext>.page_element_detections.png
    """
    stage2_json_path = stage2_json_path or _stage2_json_path_for_image(image_path)
    output_path = output_path or _out_path_for_image(image_path)

    if not stage2_json_path.exists():
        console.print(f"[yellow]Skip[/yellow] missing stage2 json: {stage2_json_path}")
        return None

    payload = read_json(stage2_json_path)
    dets = list(_iter_page_element_detections(payload))
    if not dets:
        console.print(f"[yellow]Skip[/yellow] no detections in: {stage2_json_path}")
        return None

    with Image.open(image_path) as im0:
        im = im0.convert("RGB")

    w, h = im.size
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    rendered = 0
    for d in dets:
        if d.score is not None and float(d.score) < float(min_score):
            continue
        x1, y1, x2, y2 = _bbox_xyxy_norm_to_px(d.bbox_xyxy_norm, w=w, h=h)
        color = _label_color(d.label_name)
        try:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(1, int(line_width)))
        except TypeError:
            # Pillow older API fallback: no width=
            draw.rectangle([x1, y1, x2, y2], outline=color)

        if draw_labels:
            score_s = f"{d.score:.3f}" if d.score is not None else "?"
            text = f"{d.label_name} {score_s}"
            tx, ty = x1 + 2, max(0, y1 - 12)
            # background box for readability
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            draw.rectangle([tx - 1, ty - 1, tx + tw + 3, ty + th + 1], fill=(0, 0, 0))
            draw.text((tx + 1, ty), text, fill=(255, 255, 255), font=font)

        rendered += 1

    if rendered == 0:
        console.print(f"[yellow]Skip[/yellow] all detections filtered by min_score={min_score}: {image_path}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path, format="PNG")
    return output_path


def render_page_element_detections_for_dir(
    *,
    input_dir: Path,
    overwrite: bool = False,
    min_score: float = 0.0,
    line_width: int = 3,
    draw_labels: bool = True,
    limit: Optional[int] = None,
) -> List[Path]:
    imgs = iter_images(input_dir)
    if limit is not None:
        imgs = imgs[: int(limit)]

    outputs: List[Path] = []
    skipped_existing = 0
    skipped_errors = 0
    rendered = 0
    for img_path in tqdm(imgs, desc="Render page element bboxes", unit="img"):
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        try:
            p = render_page_element_detections_for_image(
                image_path=img_path,
                output_path=out_path,
                min_score=min_score,
                line_width=line_width,
                draw_labels=draw_labels,
            )
            if p is not None:
                outputs.append(p)
                rendered += 1
        except Exception:
            skipped_errors += 1
            continue
    return outputs


@app.command("page-elements")
def render_page_elements(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .page_element_detections.png files."),
    min_score: float = typer.Option(0.0, "--min-score", help="Only draw detections with score >= min_score."),
    line_width: int = typer.Option(3, "--line-width", min=1, help="Bounding box line width in pixels."),
    draw_labels: bool = typer.Option(True, "--draw-labels/--no-draw-labels", help="Draw label + score text."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of images processed."),
) -> None:
    """
    For each image in input_dir, read adjacent stage2 JSON (<image>.page_elements_v3.json)
    and write <image>.page_element_detections.png with bounding boxes rendered.
    """
    outs = render_page_element_detections_for_dir(
        input_dir=input_dir,
        overwrite=overwrite,
        min_score=min_score,
        line_width=line_width,
        draw_labels=draw_labels,
        limit=limit,
    )
    console.print(f"[green]Done[/green] wrote={len(outs)}")
