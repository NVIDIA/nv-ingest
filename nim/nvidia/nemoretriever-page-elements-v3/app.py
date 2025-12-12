from __future__ import annotations

import asyncio
import base64
import ctypes
import ctypes.util
import io
import json
import logging
import mmap
import os
import threading
import time
from dataclasses import dataclass
from concurrent import futures
from contextlib import asynccontextmanager, contextmanager, nullcontext
from multiprocessing import shared_memory
from typing import Dict, List, Mapping, Optional, Tuple

import grpc
import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.json_format import MessageToDict
from nemotron_page_elements_v3.model import define_model
from nemotron_page_elements_v3.utils import postprocess_preds_page_element
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import prometheus_client as prom
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, start_http_server
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor

logger = logging.getLogger("nemoretriever-page-elements-v3")

OUTPUT_LABELS = [
    "table",
    "chart",
    "title",
    "infographic",
    "paragraph",
    "header_footer",
]
LABEL_REMAP = {"text": "paragraph"}
INPUT_IMAGES_NAME = "INPUT_IMAGES"
THRESHOLDS_NAME = "THRESHOLDS"
OUTPUT_NAME = "OUTPUT"
TRITON_MODEL_PREFERRED = "pipeline"
TRITON_MODEL_ALIASES = {TRITON_MODEL_PREFERRED, "yolox_ensemble", "yolox"}


class ServiceSettings(BaseSettings):
    model_name: str = "nemoretriever-page-elements-v3"
    model_version: str = Field("1.7.0", alias="YOLOX_TAG")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    auth_token: str | None = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: str | None = Field(None, alias="NIM_NGC_API_KEY")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    @property
    def resolved_auth_token(self) -> str | None:
        return self.auth_token or self.fallback_auth_token


settings = ServiceSettings()
logging.basicConfig(level=logging.DEBUG if settings.log_verbose else logging.INFO)

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    otel_metrics = None


def _configure_tracing() -> Optional["trace.Tracer"]:
    if not settings.enable_otel or not trace or not otel_metrics:
        return None

    try:
        resource = Resource.create({"service.name": settings.otel_service_name or settings.model_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_endpoint or "http://localhost:4318/v1/traces", insecure=True)
            )
        )
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=settings.otel_endpoint or "http://localhost:4318/v1/metrics", insecure=True)
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        otel_metrics.set_meter_provider(meter_provider)
        return trace.get_tracer(settings.otel_service_name or settings.model_name)
    except Exception:  # pragma: no cover - surfaced to logs
        logger.exception("Failed to configure OTEL")
        return None


TRACER = _configure_tracing()


@dataclass
class ModelState:
    loaded: bool = True
    inference_count: int = 0
    last_inference_ns: int = 0
    success_ns_total: int = 0


model_state = ModelState()


@dataclass
class SystemSharedMemoryRegion:
    name: str
    key: str
    offset: int
    byte_size: int
    shm: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: object | None = None

    @property
    def buffer(self) -> memoryview:
        if self.shm is not None:
            return self.shm.buf
        if self.mmap_obj is not None:
            return memoryview(self.mmap_obj)
        raise RuntimeError("Shared memory region has no backing buffer")

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed closing shared memory file handle", exc_info=True)


@dataclass
class CudaSharedMemoryRegion:
    name: str
    device_id: int
    byte_size: int
    raw_handle: bytes
    offset: int = 0
    pointer: int | None = None


_system_shared_memory: Dict[str, SystemSharedMemoryRegion] = {}
_cuda_shared_memory: Dict[str, CudaSharedMemoryRegion] = {}
_shared_memory_lock = threading.Lock()
_model_lock = threading.Lock()
_model_state_lock = threading.Lock()
_cudart: ctypes.CDLL | None = None


CUDA_IPC_FLAG = 1
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class ImageInput(BaseModel):
    type: str
    url: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value != "image_url":
            raise ValueError("type must be 'image_url'")
        return value


class InferRequest(BaseModel):
    input: List[ImageInput]

    @field_validator("input")
    @classmethod
    def ensure_not_empty(cls, value: List[ImageInput]) -> List[ImageInput]:
        if not value:
            raise ValueError("input must contain at least one image")
        return value


class AsyncRateLimiter:
    def __init__(self, max_concurrent: Optional[int]):
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self):
        if not self._sem:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


class SyncRateLimiter:
    def __init__(self, max_concurrent: Optional[int]):
        self._sem = threading.BoundedSemaphore(max_concurrent) if max_concurrent else None

    @contextmanager
    def limit(self):
        if not self._sem:
            yield
            return
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


REQUEST_COUNTER = prom.Counter("nim_requests_total", "Total requests served", ["protocol", "endpoint", "status"])
REQUEST_LATENCY = prom.Histogram(
    "nim_request_latency_seconds",
    "Latency for requests",
    ["protocol", "endpoint"],
    buckets=(
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    ),
)
INFLIGHT_GAUGE = prom.Gauge("nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"])

async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
sync_rate_limiter = SyncRateLimiter(settings.rate_limit)

_metrics_started = False
if settings.metrics_port and not _metrics_started:
    start_http_server(settings.metrics_port, registry=prom.REGISTRY)
    _metrics_started = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


def _is_authorized(auth_header: Optional[str]) -> bool:
    token = settings.resolved_auth_token
    if not token:
        return True
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return False
    return auth_header.split(" ", maxsplit=1)[1] == token


async def _require_http_auth(request: Request) -> None:
    if _is_authorized(request.headers.get("authorization")):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def _require_grpc_auth(context: grpc.ServicerContext) -> None:
    metadata: Mapping[str, str] = {k.lower(): v for k, v in context.invocation_metadata()}
    header = metadata.get("authorization")
    if _is_authorized(header):
        return
    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")


def _decode_image_payload(payload: str) -> np.ndarray:
    if "base64," in payload:
        _, encoded = payload.split("base64,", maxsplit=1)
    else:
        encoded = payload
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def _load_model() -> torch.nn.Module:
    model = define_model("page_element_v3", verbose=False)
    model.eval()
    return model


def _reload_model() -> torch.nn.Module:
    model = _load_model()
    app.state.model = model
    with _model_state_lock:
        model_state.loaded = True
    return model


def _unload_model() -> None:
    with _model_lock:
        app.state.model = None
    with _model_state_lock:
        model_state.loaded = False


def _resolve_thresholds(thresholds: Optional[np.ndarray], index: int) -> Tuple[Optional[float], Optional[float]]:
    if thresholds is None or thresholds.size == 0:
        return None, None
    if thresholds.ndim == 1:
        if thresholds.size == 0:
            return None, None
        return float(thresholds[min(index, thresholds.shape[0] - 1)]), None
    row = thresholds[min(index, thresholds.shape[0] - 1)]
    conf = float(row[0]) if row.size > 0 else None
    iou = float(row[1]) if row.size > 1 else None
    return conf, iou


def _convert_predictions(
    preds: List[Dict[str, torch.Tensor]], model: torch.nn.Module
) -> List[Dict[str, List[List[float]]]]:
    results: List[Dict[str, List[List[float]]]] = []
    for pred in preds:
        boxes, labels, scores = postprocess_preds_page_element(
            pred, model.thresholds_per_class, model.labels
        )

        bounding_boxes: Dict[str, List[List[float]]] = {label: [] for label in OUTPUT_LABELS}
        for box, label_idx, score in zip(boxes, labels, scores):
            label_name = model.labels[int(label_idx)]
            output_label = LABEL_REMAP.get(label_name, label_name)
            if output_label not in bounding_boxes:
                continue

            xmin, ymin, xmax, ymax = (float(coord) for coord in box.tolist())
            bounding_boxes[output_label].append([xmin, ymin, xmax, ymax, float(score)])

        results.append(bounding_boxes)
    return results


def _predict_raw(
    images: List[np.ndarray],
    model: torch.nn.Module,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, List[List[float]]]]:
    if not images:
        return []

    if thresholds is not None and thresholds.size > 0:
        results: List[Dict[str, List[List[float]]]] = []
        for idx, image in enumerate(images):
            conf_override, iou_override = _resolve_thresholds(thresholds, idx)
            original_conf = getattr(model, "conf_thresh", None)
            original_iou = getattr(model, "iou_thresh", None)
            try:
                if conf_override is not None:
                    model.conf_thresh = conf_override
                if iou_override is not None:
                    model.iou_thresh = iou_override
                processed = model.preprocess(image)
                with torch.inference_mode():
                    preds = model(processed.unsqueeze(0), np.array([image.shape[:2]]))
                results.extend(_convert_predictions(preds, model))
            finally:
                if conf_override is not None and original_conf is not None:
                    model.conf_thresh = original_conf
                if iou_override is not None and original_iou is not None:
                    model.iou_thresh = original_iou
        return results

    processed = [model.preprocess(image) for image in images]
    batch = torch.stack(processed)
    original_sizes = np.array([image.shape[:2] for image in images])

    with torch.inference_mode():
        preds = model(batch, original_sizes)

    return _convert_predictions(preds, model)


def _format_http_prediction(raw_prediction: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    return {
        "bounding_boxes": {
            label: [
                {
                    "x_min": bbox[0],
                    "y_min": bbox[1],
                    "x_max": bbox[2],
                    "y_max": bbox[3],
                    "confidence": bbox[4],
                }
                for bbox in raw_prediction.get(label, [])
            ]
            for label in OUTPUT_LABELS
        }
    }


def _serialize_predictions(raw_predictions: List[Dict[str, List[List[float]]]]) -> bytes:
    payload = [json.dumps(pred) for pred in raw_predictions]
    tensor = np.array([p.encode("utf-8") for p in payload], dtype=np.object_)
    return serialize_byte_tensor(tensor)


def _get_cudart() -> ctypes.CDLL:
    global _cudart
    if _cudart is not None:
        return _cudart
    library_path = ctypes.util.find_library("cudart")
    candidates = [library_path] if library_path else []
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.dylib",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            _cudart = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if _cudart is None:
        raise RuntimeError("CUDA runtime library not found")
    _cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    _cudart.cudaMemcpy.restype = ctypes.c_int
    _cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _cudart.cudaSetDevice.restype = ctypes.c_int
    _cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    return _cudart


def _check_cuda(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error code {result}")


def _close_cuda_pointer(pointer: int | None, device_id: int) -> None:
    if pointer is None:
        return
    try:
        cudart = _get_cudart()
        _check_cuda(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")
        _check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)), "cudaIpcCloseMemHandle")
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Failed closing CUDA shared memory handle", exc_info=True)


def _parse_shared_memory_params(
    tensor: service_pb2.ModelInferRequest.InferInputTensor,
) -> Tuple[Optional[str], Optional[int], int]:
    params = tensor.parameters
    region = params.get("shared_memory_region").string_param if "shared_memory_region" in params else None
    byte_size = params.get("shared_memory_byte_size").int64_param if "shared_memory_byte_size" in params else None
    offset = params.get("shared_memory_offset").int64_param if "shared_memory_offset" in params else 0
    return region or None, byte_size, offset


def _read_system_shared_memory(region_name: str, byte_size: int, offset: int) -> bytes:
    with _shared_memory_lock:
        if region_name not in _system_shared_memory:
            raise KeyError(region_name)
        region = _system_shared_memory[region_name]
    start = region.offset + offset
    end = start + byte_size
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    return bytes(region.buffer[start:end])


def _write_system_shared_memory(region_name: str, data: bytes, offset: int, byte_size: int) -> None:
    with _shared_memory_lock:
        if region_name not in _system_shared_memory:
            raise KeyError(region_name)
        region = _system_shared_memory[region_name]
    start = region.offset + offset
    end = start + byte_size
    if len(data) > byte_size:
        raise ValueError("Shared memory target too small for data")
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    region.buffer[start:start + len(data)] = data


def _read_cuda_shared_memory(region_name: str, byte_size: int, offset: int) -> bytes:
    with _shared_memory_lock:
        if region_name not in _cuda_shared_memory:
            raise KeyError(region_name)
        region = _cuda_shared_memory[region_name]
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + byte_size
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    cudart = _get_cudart()
    _check_cuda(cudart.cudaSetDevice(int(region.device_id)), "cudaSetDevice")
    host_buffer = (ctypes.c_char * byte_size)()
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda(
        cudart.cudaMemcpy(
            ctypes.cast(host_buffer, ctypes.c_void_p),
            device_ptr,
            ctypes.c_size_t(byte_size),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "cudaMemcpy (device to host)",
    )
    return bytes(host_buffer)


def _write_cuda_shared_memory(region_name: str, data: bytes, offset: int, byte_size: int) -> None:
    with _shared_memory_lock:
        if region_name not in _cuda_shared_memory:
            raise KeyError(region_name)
        region = _cuda_shared_memory[region_name]
    if not data:
        return
    if len(data) > byte_size:
        raise ValueError("Shared memory target too small for data")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + len(data)
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    cudart = _get_cudart()
    _check_cuda(cudart.cudaSetDevice(int(region.device_id)), "cudaSetDevice")
    host_buffer = ctypes.create_string_buffer(data)
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda(
        cudart.cudaMemcpy(
            device_ptr,
            ctypes.cast(host_buffer, ctypes.c_void_p),
            ctypes.c_size_t(len(data)),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "cudaMemcpy (host to device)",
    )


def _register_system_shared_memory(name: str, key: str, offset: int, byte_size: int) -> None:
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    with _shared_memory_lock:
        if name in _system_shared_memory or name in _cuda_shared_memory:
            raise FileExistsError(f"Region {name} already registered")
    shm_obj: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: object | None = None
    buffer_size: int
    try:
        try:
            shm_obj = shared_memory.SharedMemory(name=key, create=False)
            buffer_size = shm_obj.size
        except FileNotFoundError:
            path = key[7:] if key.startswith("file://") else key
            file_handle = open(path, "r+b")
            file_handle.seek(0, os.SEEK_END)
            buffer_size = file_handle.tell()
            file_handle.seek(0)
            mmap_obj = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_WRITE)
        if offset < 0 or offset + byte_size > buffer_size:
            raise ValueError("Shared memory region too small for requested window")
        region = SystemSharedMemoryRegion(
            name=name, key=key, offset=offset, byte_size=byte_size, shm=shm_obj, mmap_obj=mmap_obj, file_handle=file_handle
        )
        with _shared_memory_lock:
            _system_shared_memory[name] = region
    except Exception:
        if shm_obj is not None:
            shm_obj.close()
        if mmap_obj is not None:
            mmap_obj.close()
        if file_handle is not None:
            file_handle.close()
        raise


def _unregister_system_shared_memory(name: str) -> None:
    with _shared_memory_lock:
        region = _system_shared_memory.pop(name, None)
    if region:
        region.close()


def _register_cuda_shared_memory(name: str, raw_handle: bytes, device_id: int, byte_size: int) -> None:
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    if not raw_handle:
        raise ValueError("raw_handle is required for CUDA shared memory registration")
    with _shared_memory_lock:
        if name in _cuda_shared_memory or name in _system_shared_memory:
            raise FileExistsError(f"Region {name} already registered")
    pointer: int | None = None
    cudart: ctypes.CDLL | None = None
    try:
        cudart = _get_cudart()
        _check_cuda(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")
        handle_buffer = ctypes.create_string_buffer(raw_handle)
        device_pointer = ctypes.c_void_p()
        _check_cuda(
            cudart.cudaIpcOpenMemHandle(
                ctypes.byref(device_pointer),
                ctypes.cast(handle_buffer, ctypes.c_void_p),
                CUDA_IPC_FLAG,
            ),
            "cudaIpcOpenMemHandle",
        )
        if device_pointer.value is None:
            raise RuntimeError("cudaIpcOpenMemHandle returned null pointer")
        pointer = int(device_pointer.value)
        region = CudaSharedMemoryRegion(
            name=name,
            device_id=int(device_id),
            byte_size=int(byte_size),
            raw_handle=bytes(raw_handle),
            pointer=pointer,
        )
        with _shared_memory_lock:
            if name in _cuda_shared_memory or name in _system_shared_memory:
                raise FileExistsError(f"Region {name} already registered")
            _cuda_shared_memory[name] = region
    except Exception:
        if pointer is not None and cudart is not None:
            try:
                _check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)), "cudaIpcCloseMemHandle")
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed closing CUDA IPC handle after registration error", exc_info=True)
        raise


def _unregister_cuda_shared_memory(name: str) -> None:
    with _shared_memory_lock:
        region = _cuda_shared_memory.pop(name, None)
    if region is not None:
        _close_cuda_pointer(region.pointer, region.device_id)


app = FastAPI(title="nemoretriever-page-elements-v3", version=settings.model_version)
app.state.model = _load_model()


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    return response


@app.get("/v1/health/live")
@app.get("/v2/health/live")
def live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v1/health/ready")
@app.get("/v2/health/ready")
def ready() -> Dict[str, bool]:
    return {"ready": model_state.loaded and app.state.model is not None}


@app.get("/v1/metadata")
async def metadata(_: None = Depends(_require_http_auth)) -> Dict[str, object]:
    short_name = f"{settings.model_name}:{settings.model_version}"
    return {
        "id": settings.model_name,
        "name": settings.model_name,
        "version": settings.model_version,
        "modelInfo": [
            {
                "name": settings.model_name,
                "version": settings.model_version,
                "shortName": short_name,
                "maxBatchSize": settings.max_batch_size,
            }
        ],
    }


@app.get("/metrics")
@app.get("/v2/metrics")
def metrics(_: None = Depends(_require_http_auth)) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _record_metrics(endpoint: str, protocol: str, status: str, start_time: float) -> None:
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(time.perf_counter() - start_time)


@app.post("/v1/infer")
async def infer(
    request_body: InferRequest, _: None = Depends(_require_http_auth)
) -> Dict[str, List[Dict[str, Dict[str, List[Dict[str, float]]]]]]:
    start_time = time.perf_counter()
    endpoint = "/v1/infer"
    if len(request_body.input) > settings.max_batch_size:
        _record_metrics(endpoint, "http", "batch_limit", start_time)
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request_body.input)} exceeds limit {settings.max_batch_size}",
        )
    if not model_state.loaded or app.state.model is None:
        _record_metrics(endpoint, "http", "not_loaded", start_time)
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        images = [_decode_image_payload(item.url) for item in request_body.input]
    except Exception as exc:
        logger.error("Failed to decode input image: %s", exc)
        _record_metrics(endpoint, "http", "invalid_payload", start_time)
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc

    try:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).inc()
        tracer_ctx = TRACER.start_as_current_span("http_infer") if TRACER else nullcontext()
        with tracer_ctx:
            async with async_rate_limiter.limit():
                with _model_lock:
                    raw_predictions = _predict_raw(images, app.state.model, None)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        _record_metrics(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).dec()

    predictions = [_format_http_prediction(prediction) for prediction in raw_predictions]
    _record_metrics(endpoint, "http", "success", start_time)
    with _model_state_lock:
        model_state.inference_count += 1
        model_state.last_inference_ns = time.time_ns()
        model_state.success_ns_total += int((time.perf_counter() - start_time) * 1e9)
    return {"data": predictions}


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "nemoretriever-page-elements-v3 HTTP"}, status_code=200)


@app.get("/v2/models/{model_name}")
async def model_metadata_http(model_name: str, _: None = Depends(_require_http_auth)) -> JSONResponse:
    normalized = _normalize_model_name(model_name)
    response = _model_metadata_response(normalized)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/config")
async def model_config_http(model_name: str, _: None = Depends(_require_http_auth)) -> JSONResponse:
    if not model_state.loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    normalized = _normalize_model_name(model_name)
    response = _model_config_response(normalized)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> JSONResponse:
    _ = _normalize_model_name(model_name)
    return JSONResponse({"ready": model_state.loaded and app.state.model is not None})


def _normalize_model_name(model_name: Optional[str]) -> str:
    if model_name in TRITON_MODEL_ALIASES:
        return model_name  # keep the alias provided by the caller
    return TRITON_MODEL_PREFERRED


def _model_ready_response() -> service_pb2.ModelReadyResponse:
    return service_pb2.ModelReadyResponse(ready=model_state.loaded and app.state.model is not None)


def _server_ready_response() -> service_pb2.ServerReadyResponse:
    return service_pb2.ServerReadyResponse(ready=model_state.loaded and app.state.model is not None)


def _server_live_response() -> service_pb2.ServerLiveResponse:
    return service_pb2.ServerLiveResponse(live=True)


def _model_metadata_response(model_name: str) -> service_pb2.ModelMetadataResponse:
    tensor_inputs = [
        service_pb2.ModelMetadataResponse.TensorMetadata(
            name=INPUT_IMAGES_NAME, datatype="BYTES", shape=[-1]
        ),
        service_pb2.ModelMetadataResponse.TensorMetadata(
            name=THRESHOLDS_NAME, datatype="FP32", shape=[-1, 2]
        ),
    ]
    tensor_outputs = [
        service_pb2.ModelMetadataResponse.TensorMetadata(
            name=OUTPUT_NAME, datatype="BYTES", shape=[-1]
        )
    ]
    return service_pb2.ModelMetadataResponse(
        name=model_name,
        versions=[settings.model_version],
        platform="python",
        inputs=tensor_inputs,
        outputs=tensor_outputs,
    )


def _model_config_response(model_name: str) -> model_config_pb2.ModelConfigResponse:
    config = model_config_pb2.ModelConfig(
        name=model_name,
        platform="python",
        max_batch_size=settings.max_batch_size,
        input=[
            model_config_pb2.ModelInput(name=INPUT_IMAGES_NAME, data_type=model_config_pb2.TYPE_BYTES, dims=[-1]),
            model_config_pb2.ModelInput(name=THRESHOLDS_NAME, data_type=model_config_pb2.TYPE_FP32, dims=[-1, 2]),
        ],
        output=[
            model_config_pb2.ModelOutput(name=OUTPUT_NAME, data_type=model_config_pb2.TYPE_BYTES, dims=[-1]),
        ],
    )
    return model_config_pb2.ModelConfigResponse(config=config)


def _model_repository_index() -> service_pb2.RepositoryIndexResponse:
    loaded = model_state.loaded and app.state.model is not None
    models = [
        service_pb2.RepositoryIndexResponse.ModelIndex(
            name=model_name,
            version=settings.model_version,
            state="READY" if loaded else "UNAVAILABLE",
            reason="" if loaded else "Model is unloaded",
        )
        for model_name in sorted(TRITON_MODEL_ALIASES)
    ]
    return service_pb2.RepositoryIndexResponse(models=models)


def _deserialize_bytes_input(
    tensor: service_pb2.ModelInferRequest.InferInputTensor, raw_contents: Optional[bytes]
) -> List[str]:
    if raw_contents:
        values = deserialize_bytes_tensor(raw_contents)
    elif tensor.contents.raw_contents:
        values = deserialize_bytes_tensor(tensor.contents.raw_contents)
    elif tensor.contents.bytes_contents:
        values = np.array(list(tensor.contents.bytes_contents), dtype=np.object_)
    else:
        values = np.array([], dtype=np.object_)
    flattened = values.reshape(-1)
    return [
        item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        for item in flattened.tolist()
    ]


def _deserialize_fp32_input(
    tensor: service_pb2.ModelInferRequest.InferInputTensor, raw_contents: Optional[bytes]
) -> np.ndarray:
    if raw_contents:
        return np.frombuffer(raw_contents, dtype=np.float32).reshape(tuple(tensor.shape) or (-1,))
    if tensor.contents.raw_contents:
        return np.frombuffer(tensor.contents.raw_contents, dtype=np.float32).reshape(tuple(tensor.shape) or (-1,))
    if tensor.contents.fp32_contents:
        return np.array(list(tensor.contents.fp32_contents), dtype=np.float32).reshape(tuple(tensor.shape) or (-1,))
    return np.empty((0,), dtype=np.float32)


class TritonService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, state: ModelState):
        super().__init__()
        self._state = state

    def ServerLive(self, request, context):
        return _server_live_response()

    def ServerReady(self, request, context):
        return _server_ready_response()

    def ModelReady(self, request, context):
        return _model_ready_response()

    def ServerMetadata(self, request, context):
        return service_pb2.ServerMetadataResponse(
            name=settings.model_name,
            version=settings.model_version,
            extensions=["model_repository", "metrics"],
        )

    def ModelMetadata(self, request, context):
        _require_grpc_auth(context)
        model_name = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        return _model_metadata_response(model_name)

    def ModelConfig(self, request, context):
        _require_grpc_auth(context)
        if not model_state.loaded or app.state.model is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        model_name = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        return _model_config_response(model_name)

    def RepositoryIndex(self, request, context):
        return _model_repository_index()

    def RepositoryModelLoad(self, request, context):
        _require_grpc_auth(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        with _model_lock:
            if app.state.model is None:
                _reload_model()
            else:
                with _model_state_lock:
                    self._state.loaded = True
        return service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(self, request, context):
        _require_grpc_auth(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        with _model_state_lock:
            self._state.loaded = False
        _unload_model()
        return service_pb2.RepositoryModelUnloadResponse()

    def _infer_once(self, request, context, endpoint: str) -> service_pb2.ModelInferResponse:
        start_time = time.perf_counter()
        _require_grpc_auth(context)
        if not self._state.loaded or app.state.model is None:
            _record_metrics(endpoint, "grpc", "not_loaded", start_time)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        model_name = _normalize_model_name(request.model_name or TRITON_MODEL_PREFERRED)
        try:
            input_lookup: Dict[str, int] = {inp.name: idx for idx, inp in enumerate(request.inputs)}
            if INPUT_IMAGES_NAME not in input_lookup:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INPUT_IMAGES missing")
            image_index = input_lookup[INPUT_IMAGES_NAME]
            thresholds_index = input_lookup.get(THRESHOLDS_NAME)

            def _raw_for_tensor(tensor: service_pb2.ModelInferRequest.InferInputTensor, index: int) -> Optional[bytes]:
                if index < len(request.raw_input_contents):
                    return request.raw_input_contents[index]
                region_name, region_size, region_offset = _parse_shared_memory_params(tensor)
                if region_name is None:
                    return None
                if region_size is None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required")
                try:
                    with _shared_memory_lock:
                        is_cuda = region_name in _cuda_shared_memory
                        is_system = region_name in _system_shared_memory
                    if not is_cuda and not is_system:
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {region_name}")
                    if is_cuda:
                        return _read_cuda_shared_memory(region_name, int(region_size), int(region_offset))
                    return _read_system_shared_memory(region_name, int(region_size), int(region_offset))
                except KeyError:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {region_name}")
                except ValueError as exc:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except RuntimeError as exc:
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

            image_tensor = request.inputs[image_index]
            image_raw = _raw_for_tensor(image_tensor, image_index)
            image_strings = _deserialize_bytes_input(image_tensor, image_raw)

            threshold_array = None
            if thresholds_index is not None:
                threshold_tensor = request.inputs[thresholds_index]
                threshold_raw = _raw_for_tensor(threshold_tensor, thresholds_index)
                threshold_array = _deserialize_fp32_input(threshold_tensor, threshold_raw)

            flat_images = [s for s in image_strings if s]
            batch_size = len(flat_images)
            if batch_size == 0:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No images provided")
            if batch_size > settings.max_batch_size:
                _record_metrics(endpoint, "grpc", "batch_limit", start_time)
                context.abort(
                    grpc.StatusCode.OUT_OF_RANGE,
                    f"Batch size {batch_size} exceeds limit {settings.max_batch_size}",
                )

            images: List[np.ndarray] = [_decode_image_payload(value) for value in flat_images]

            output_region: Optional[str] = None
            output_offset = 0
            output_byte_size: Optional[int] = None
            output_is_cuda = False
            if request.outputs:
                output_tensor = request.outputs[0]
                region_name, region_size, region_offset = _parse_shared_memory_params(output_tensor)
                if region_name:
                    with _shared_memory_lock:
                        output_is_cuda = region_name in _cuda_shared_memory
                        output_is_system = region_name in _system_shared_memory
                    if not output_is_cuda and not output_is_system:
                        _record_metrics(endpoint, "grpc", "error", start_time)
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {region_name}")
                    if region_size is None:
                        _record_metrics(endpoint, "grpc", "error", start_time)
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required for outputs")
                    output_region = region_name
                    output_offset = int(region_offset)
                    output_byte_size = int(region_size)

            INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).inc()
            try:
                tracer_ctx = TRACER.start_as_current_span("grpc_infer") if TRACER else nullcontext()
                with tracer_ctx:
                    with sync_rate_limiter.limit():
                        with _model_lock:
                            raw_predictions = _predict_raw(images, app.state.model, threshold_array)
            finally:
                INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).dec()

            serialized = _serialize_predictions(raw_predictions)
            output = service_pb2.ModelInferResponse.InferOutputTensor(
                name=OUTPUT_NAME, datatype="BYTES", shape=[batch_size]
            )
            response = service_pb2.ModelInferResponse(
                model_name=model_name,
                outputs=[output],
                id=request.id,
            )
            if output_region is not None and output_byte_size is not None:
                try:
                    if output_is_cuda:
                        _write_cuda_shared_memory(output_region, serialized, output_offset, output_byte_size)
                    else:
                        _write_system_shared_memory(output_region, serialized, output_offset, output_byte_size)
                except KeyError:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {output_region}")
                except ValueError as exc:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except RuntimeError as exc:
                    logger.exception("Failed to write CUDA shared memory output: %s", exc)
                    _record_metrics(endpoint, "grpc", "error", start_time)
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
                except Exception as exc:
                    logger.exception("Failed to write output to shared memory: %s", exc)
                    _record_metrics(endpoint, "grpc", "error", start_time)
                    context.abort(grpc.StatusCode.INTERNAL, "Failed to write shared memory output")
                response.raw_output_contents[:] = []
            else:
                response.raw_output_contents.append(serialized)
            _record_metrics(endpoint, "grpc", "success", start_time)
            with _model_state_lock:
                self._state.inference_count += 1
                self._state.last_inference_ns = time.time_ns()
                self._state.success_ns_total += int((time.perf_counter() - start_time) * 1e9)
            return response
        except grpc.RpcError:
            _record_metrics(endpoint, "grpc", "error", start_time)
            raise
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("gRPC inference failed: %s", exc)
            _record_metrics(endpoint, "grpc", "error", start_time)
            context.abort(grpc.StatusCode.INTERNAL, "Inference failed")

    def ModelInfer(self, request, context):
        return self._infer_once(request, context, "grpc::infer")

    def ModelStatistics(self, request, context):
        _require_grpc_auth(context)
        model_name = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        stats = service_pb2.ModelStatistics(
            name=model_name,
            version=settings.model_version,
            last_inference=self._state.last_inference_ns,
            inference_count=self._state.inference_count,
            execution_count=self._state.inference_count,
        )
        stats.inference_stats.success.count = self._state.inference_count
        stats.inference_stats.success.ns = self._state.success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    def ModelStreamInfer(self, request_iterator, context):
        for infer_request in request_iterator:
            yield self._infer_once(infer_request, context, "grpc::stream_infer")

    def SystemSharedMemoryStatus(self, request, context):
        _require_grpc_auth(context)
        response = service_pb2.SystemSharedMemoryStatusResponse()
        with _shared_memory_lock:
            if request.name:
                if request.name not in _system_shared_memory:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown region {request.name}")
                regions = [request.name]
            else:
                regions = list(_system_shared_memory.keys())
            for region_name in regions:
                region = _system_shared_memory[region_name]
                status = service_pb2.SystemSharedMemoryStatusResponse.RegionStatus(
                    name=region.name,
                    key=region.key,
                    offset=region.offset,
                    byte_size=region.byte_size,
                )
                response.regions[region.name].CopyFrom(status)
        return response

    def SystemSharedMemoryRegister(self, request, context):
        _require_grpc_auth(context)
        if not request.name or not request.key:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name and key are required")
        try:
            _register_system_shared_memory(request.name, request.key, int(request.offset), int(request.byte_size))
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("System shared memory registration failed: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to register shared memory")
        return service_pb2.SystemSharedMemoryRegisterResponse()

    def SystemSharedMemoryUnregister(self, request, context):
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        with _shared_memory_lock:
            if request.name not in _system_shared_memory:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown region {request.name}")
        _unregister_system_shared_memory(request.name)
        return service_pb2.SystemSharedMemoryUnregisterResponse()

    def CudaSharedMemoryStatus(self, request, context):
        _require_grpc_auth(context)
        response = service_pb2.CudaSharedMemoryStatusResponse()
        with _shared_memory_lock:
            if request.name:
                if request.name not in _cuda_shared_memory:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown CUDA region {request.name}")
                regions = [request.name]
            else:
                regions = list(_cuda_shared_memory.keys())
            for region_name in regions:
                region = _cuda_shared_memory[region_name]
                status = service_pb2.CudaSharedMemoryStatusResponse.RegionStatus(
                    name=region.name,
                    device_id=region.device_id,
                    byte_size=region.byte_size,
                )
                response.regions[region.name].CopyFrom(status)
        return response

    def CudaSharedMemoryRegister(self, request, context):
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            _register_cuda_shared_memory(
                request.name,
                bytes(request.raw_handle),
                int(request.device_id),
                int(request.byte_size),
            )
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return service_pb2.CudaSharedMemoryRegisterResponse()

    def CudaSharedMemoryUnregister(self, request, context):
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        with _shared_memory_lock:
            if request.name not in _cuda_shared_memory:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown CUDA region {request.name}")
        _unregister_cuda_shared_memory(request.name)
        return service_pb2.CudaSharedMemoryUnregisterResponse()


def _start_grpc_server() -> grpc.Server:
    grpc_workers = max(settings.rate_limit or 4, 4)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(TritonService(model_state), server)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    server.start()
    logger.info("gRPC server listening on %s", settings.grpc_port)
    return server


grpc_server: grpc.Server | None = None


@app.on_event("startup")
def _startup() -> None:
    global grpc_server
    if grpc_server is None:
        grpc_server = _start_grpc_server()


@app.on_event("shutdown")
def _shutdown() -> None:
    global grpc_server
    if grpc_server:
        grpc_server.stop(grace=0)
        grpc_server = None
    _unload_model()
