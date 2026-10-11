"""FusedGPUOperator + device-aware example operators (P3').

The point: existing operators exchange data as pandas DataFrames across Ray ``map_batches``
boundaries, where any device tensor would be serialized to host (and today images travel as
base64). A *fused* operator runs a list of operators IN ONE PROCESS, threading the DataFrame
through each child's ``preprocess → process → postprocess``. Intermediate device tensors
(``DeviceImage``, DLPack-exportable) then survive between operators with ZERO host copies.

This keeps the **exact existing operator API** — children are ordinary ``AbstractOperator``s.
A child becomes "device-aware" simply by reading/writing the ``page_image_dev`` column
(a column of ``DeviceImage`` handles) instead of base64. Non-fused/legacy operators that only
know base64 still work unchanged; they just don't get the zero-copy benefit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]            # gpu_pdf_extractor/
sys.path.insert(0, str(_ROOT / "python"))              # gpu_pdfium package
sys.path.insert(0, str(_ROOT / "native" / "build"))    # _gpu_raster, _gpu_pdfium
sys.path.insert(0, str(_ROOT.parent / "nemo_retriever" / "src"))  # real operator base

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator

import gpu_pdfium                       # PDFium-backed renderer (P1)
import _gpu_raster as _raster           # CUDA DLPack device handoff (P3')

DEVICE_COL = "page_image_dev"           # column carrying DeviceImage handles between fused ops


class FusedGPUOperator(AbstractOperator, GPUOperator):
    """Run a list of operators in one process, end to end.

    For each child it invokes ``preprocess → process → postprocess`` explicitly (the same calls
    ``AbstractOperator.run`` makes), passing each child's output as the next child's input. Because
    everything runs in a single process / CUDA context, device tensors placed on the batch (e.g.
    the ``page_image_dev`` column) are handed off zero-copy — no Ray serialization, no base64,
    no device↔host round-trip between stages.
    """

    def __init__(self, operators: List[AbstractOperator] | None = None,
                 operator_specs: List[tuple] | None = None, **kwargs: Any) -> None:
        # `operator_specs` (list of (OperatorClass, kwargs_dict)) is the Ray-friendly path: it is
        # picklable (classes by reference + plain dicts) and the children — which hold un-picklable
        # torch models — are constructed HERE, i.e. on the Ray worker. `operators` (pre-built) is for
        # in-process use. Only operator_specs is captured for reconstruction (see get_constructor_kwargs).
        super().__init__(operator_specs=operator_specs, **kwargs)
        if operators is not None:
            self.operators = list(operators)
        else:
            self.operators = [cls(**(kw or {})) for cls, kw in (operator_specs or [])]

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        for op in self.operators:
            data = op.preprocess(data, **kwargs)
            data = op.process(data, **kwargs)
            data = op.postprocess(data, **kwargs)
        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        return self.run(data, **kwargs)

    def get_constructor_kwargs(self) -> dict:
        """Ray reconstructs from operator_specs only — never the (un-picklable) built operators."""
        kw = dict(getattr(self, "_graph_init_kwargs", {}))
        kw["operator_specs"] = getattr(self, "operator_specs", None)
        kw.pop("operators", None)
        return kw


def _fit_scale(w: float, h: float, target: int) -> float:
    if w <= 0 or h <= 0:
        return 1.0
    return max(min(target / w, target / h), 1e-3)


class RasterizeGPUOperator(AbstractOperator, GPUOperator):
    """Render each single-page PDF (column ``bytes``) and place the raster ON THE GPU.

    Uses the P1 PDFium backend to rasterize (CPU), does the single unavoidable H2D upload, and
    attaches a ``DeviceImage`` per row in the ``page_image_dev`` column. No base64 produced.
    (A future on-device CUDA rasterizer would remove even the H2D.)
    """

    def __init__(self, target_px: int = 1024, dev: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.target_px = int(target_px)
        self.dev = int(dev)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame):
            raise NotImplementedError("RasterizeGPUOperator expects a pandas DataFrame")
        dev_imgs: list = []
        for _, row in data.iterrows():
            pdf_bytes = row["bytes"]
            doc = gpu_pdfium.PdfDocument(pdf_bytes)
            try:
                page = doc.get_page(0)
                scale = _fit_scale(page.get_width(), page.get_height(), self.target_px)
                arr = page.render(scale=scale).to_numpy()          # host BGR (H,W,3)
                arr = arr if arr.flags["C_CONTIGUOUS"] else arr.copy()
                dev_imgs.append(_raster.upload_to_device(arr, self.dev))  # -> GPU (one H2D)
                page.close()
            finally:
                doc.close()
        out = data.copy()
        out[DEVICE_COL] = dev_imgs
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _dev_to_chw_rgb(dev_img):
    """DeviceImage (HWC BGR uint8) -> torch CHW RGB float32, all on GPU (zero host copy)."""
    import torch
    t = torch.from_dlpack(dev_img)                 # HWC BGR uint8, same CUDA buffer
    return t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous().to(torch.float32), int(t.shape[0]), int(t.shape[1])


class PageElementGPUOperator(AbstractOperator, GPUOperator):
    """REAL NemotronPageElementsV3 page-element detection, consuming the device raster zero-copy.

    Reads ``page_image_dev`` (DeviceImage), wraps it with ``torch.from_dlpack`` (no copy), runs the
    model on-device, and writes per-row detection boxes. This is the in-process model path
    (no remote NIM) required for zero-copy.
    """

    def __init__(self, dev: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dev = int(dev)
        from nemo_retriever.models.local import NemotronPageElementsV3
        self._model = NemotronPageElementsV3()

    def preprocess(self, data, **k): return data

    def process(self, data, **k):
        import torch
        boxes_col, labels_col = [], []
        for _, row in data.iterrows():
            chw, H, W = _dev_to_chw_rgb(row[DEVICE_COL])
            x = self._model.preprocess(chw)
            with torch.inference_mode():
                preds = self._model(x, (H, W))
            bx = lbl = None
            try:
                b, l, _s = self._model.postprocess(preds)
                b0 = b[0] if isinstance(b, (list, tuple)) else b
                l0 = l[0] if isinstance(l, (list, tuple)) else l
                bx = b0.detach().float().cpu().numpy().reshape(-1, 4) if hasattr(b0, "detach") else None
                lbl = l0.detach().cpu().numpy().reshape(-1).astype(int) if hasattr(l0, "detach") else None
            except Exception:
                pass
            boxes_col.append(bx); labels_col.append(lbl)
        out = data.copy(); out["pe_boxes"] = boxes_col; out["pe_labels"] = labels_col
        return out

    def postprocess(self, data, **k): return data


class CropGPUOperator(AbstractOperator, GPUOperator):
    """Crop detected regions from the device page tensor ON-DEVICE (torch slicing stays on GPU).

    Produces a per-row list of device crop tensors that never leave the GPU — ready to feed a
    downstream table-structure / OCR model in the same process (the next operator).
    """

    # page-element class index -> label name
    LABELS = ["table", "chart", "title", "infographic", "text", "header_footer"]

    def process(self, data, **k):
        import torch
        crops_col = []
        for _, row in data.iterrows():
            t = torch.from_dlpack(row[DEVICE_COL])      # HWC BGR uint8 on GPU (same buffer)
            H, W = int(t.shape[0]), int(t.shape[1])
            boxes = row.get("pe_boxes")
            labels = row.get("pe_labels")
            crops = []
            if boxes is not None:
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    # model boxes are normalized [0,1] -> scale to raster pixels
                    xa, ya = max(0, int(x1 * W)), max(0, int(y1 * H))
                    xb, yb = min(W, int(x2 * W)), min(H, int(y2 * H))
                    if xb - xa >= 4 and yb - ya >= 4:
                        li = int(labels[i]) if labels is not None and i < len(labels) else -1
                        name = self.LABELS[li] if 0 <= li < len(self.LABELS) else "?"
                        crops.append({"label": name, "t": t[ya:yb, xa:xb, :]})  # on-device slice
            crops_col.append(crops)
        out = data.copy(); out["region_crops_dev"] = crops_col
        return out

    def preprocess(self, d, **k): return d
    def postprocess(self, d, **k): return d


class TableStructureGPUOperator(AbstractOperator, GPUOperator):
    """REAL NemotronTableStructureV1 on the on-device 'table' crops (no host round-trip).

    Each table crop is already a CUDA tensor (a slice of the page buffer); we convert to the model's
    RGB CHW input on-device and run the real model. Proves the full rasterize→detect→crop→table
    chain stays resident on one GPU in one process.
    """

    def __init__(self, dev: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dev = int(dev)
        from nemo_retriever.models.local import NemotronTableStructureV1
        self._model = NemotronTableStructureV1()

    def preprocess(self, d, **k): return d

    def process(self, data, **k):
        import torch
        n_tables, all_cuda = [], True
        for _, row in data.iterrows():
            cnt = 0
            for crop in (row.get("region_crops_dev") or []):
                if crop["label"] != "table":
                    continue
                t = crop["t"]                                  # HWC BGR uint8, CUDA
                all_cuda = all_cuda and bool(t.is_cuda)
                h, w = int(t.shape[0]), int(t.shape[1])
                chw = t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous().to(torch.float32)
                x = self._model.preprocess(chw, (h, w))
                with torch.inference_mode():
                    _ = self._model.invoke(x, (h, w))          # real table-structure inference
                cnt += 1
            n_tables.append(cnt)
        out = data.copy(); out["n_tables_structured"] = n_tables
        out.attrs["table_inputs_on_device"] = all_cuda
        return out

    def postprocess(self, d, **k): return d


class OCRStubGPUOperator(AbstractOperator, GPUOperator):
    """Stand-in for table-structure / OCR (weights not cached offline). Consumes the device crop
    tensors ON-DEVICE, proving they never round-tripped to host. Real model: torch model(crop)."""

    def process(self, data, **k):
        import torch
        all_cuda, n_crops, acts = True, [], []
        for _, row in data.iterrows():
            crops = row.get("region_crops_dev") or []
            n_crops.append(len(crops))
            for c in crops:
                all_cuda = all_cuda and bool(c.is_cuda)
                acts.append(float(c.float().mean()))     # on-device reduction (stub for OCR/table)
        out = data.copy()
        out["n_region_crops"] = n_crops
        out.attrs["all_crops_on_device"] = all_cuda
        out.attrs["mean_activations"] = acts[:8]
        return out

    def preprocess(self, d, **k): return d
    def postprocess(self, d, **k): return d


class GraphicElementsGPUOperator(AbstractOperator, GPUOperator):
    """REAL NemotronGraphicElementsV1 on the on-device 'chart' crops (zero-copy handoff)."""

    def __init__(self, dev: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dev = int(dev)
        from nemo_retriever.models.local import NemotronGraphicElementsV1
        self._model = NemotronGraphicElementsV1()

    def preprocess(self, d, **k): return d

    def process(self, data, **k):
        import torch
        n_charts, all_cuda = [], True
        for _, row in data.iterrows():
            cnt = 0
            for crop in (row.get("region_crops_dev") or []):
                if crop["label"] != "chart":
                    continue
                t = crop["t"]; all_cuda = all_cuda and bool(t.is_cuda)
                h, w = int(t.shape[0]), int(t.shape[1])
                chw = t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous().to(torch.float32)
                x = self._model.preprocess(chw)
                with torch.inference_mode():
                    _ = self._model.invoke(x, (h, w))
                cnt += 1
            n_charts.append(cnt)
        out = data.copy(); out["n_charts_structured"] = n_charts
        out.attrs["chart_inputs_on_device"] = all_cuda
        return out

    def postprocess(self, d, **k): return d


class OCRGPUOperator(AbstractOperator, GPUOperator):
    """REAL NemotronOCRV2 on the on-device 'text' crops.

    The crop is handed to the model as a CUDA CHW tensor (zero-copy up to the model); note the OCR-v2
    wrapper itself re-encodes to PNG internally (its pipeline operates on image bytes) — that's a
    property of that model, not of our operator handoff. Caps regions/page to keep latency bounded.
    """

    def __init__(self, dev: int = 0, max_regions_per_page: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dev = int(dev)
        self.max_regions_per_page = int(max_regions_per_page)
        from nemo_retriever.models.local import NemotronOCRV2
        self._model = NemotronOCRV2()

    def preprocess(self, d, **k): return d

    def process(self, data, **k):
        import torch
        n_text, chars = [], []
        for _, row in data.iterrows():
            text_crops = [c for c in (row.get("region_crops_dev") or []) if c["label"] == "text"]
            cnt = total = 0
            for crop in text_crops[: self.max_regions_per_page]:
                t = crop["t"]
                chw = t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous().to(torch.uint8)  # CHW RGB
                try:
                    res = self._model.invoke(chw)
                    txt = type(self._model)._extract_text(res)
                    total += len(txt or "")
                    cnt += 1
                except Exception:
                    pass
            n_text.append(cnt); chars.append(total)
        out = data.copy(); out["n_text_ocr"] = n_text; out["ocr_chars"] = chars
        return out

    def postprocess(self, d, **k): return d


class HostFinalizeOperator(AbstractOperator):
    """Terminal op: drop device-only columns so the fused STAGE output is host-serializable.

    Device tensors (page_image_dev / region_crops_dev) live only inside the fused actor; they cannot
    cross a Ray output boundary. This records counts, then strips the un-serializable columns so
    ds.map_batches can return the batch. CPU op (no GPU mixin)."""

    DEVICE_COLS = ("page_image_dev", "region_crops_dev")
    HEAVY_COLS = ("pe_boxes", "pe_labels")

    def preprocess(self, d, **k): return d

    def process(self, data, **k):
        out = data.copy()
        if "region_crops_dev" in out.columns and "n_region_crops" not in out.columns:
            out["n_region_crops"] = out["region_crops_dev"].apply(lambda c: len(c) if c is not None else 0)
        drop = [c for c in (*self.DEVICE_COLS, *self.HEAVY_COLS) if c in out.columns]
        return out.drop(columns=drop)

    def postprocess(self, d, **k): return d


class PageElementStubGPUOperator(AbstractOperator, GPUOperator):
    """Stand-in for the page-element / OCR GPU model: consumes the device raster ZERO-COPY.

    Reads the ``page_image_dev`` DeviceImage and runs a CUDA kernel on the SAME device buffer
    (here a byte-mean reduction; in production this is where ``torch.from_dlpack(dev_img)`` would
    feed NemotronPageElementsV3 / OCR directly). Records the device pointer it actually read so
    the demo can prove it matches the producer's pointer (no copy crossed the boundary).
    """

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if DEVICE_COL not in data.columns:
            raise RuntimeError(f"expected '{DEVICE_COL}' from an upstream device-aware operator")
        consumed_ptr, means = [], []
        for _, row in data.iterrows():
            dev_img = row[DEVICE_COL]
            # ---- production hook ----------------------------------------------------------
            # import torch; t = torch.from_dlpack(dev_img)        # zero-copy CUDA tensor
            # detections = page_elements_model(t)                 # in-process, on-device
            # -------------------------------------------------------------------------------
            res = _raster.consume_mean(dev_img)                   # kernel on the same buffer
            consumed_ptr.append(res["device_ptr"])
            means.append(res["mean"])
        out = data.copy()
        out["consumed_device_ptr"] = consumed_ptr
        out["stub_activation"] = means
        return out

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
