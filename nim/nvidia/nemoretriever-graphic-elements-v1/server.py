from __future__ import annotations

import asyncio
import base64
import binascii
import ctypes
import ctypes.util
import io
import json
import logging
import threading
import time
from concurrent import futures
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import grpc
import numpy as np
import requests
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.json_format import MessageToDict
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, HttpUrl, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor

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

try:
    from nemotron_graphic_elements_v1.model import define_model
    from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element
except ImportError as exc:
    msg = (
        "nemotron_graphic_elements_v1 is missing. "
        "Install the Hugging Face repo (git lfs install && git clone https://huggingface.co/nvidia/nemotron-graphic-elements-v1 && pip install -e ./nemotron-graphic-elements-v1)."
    )
    raise RuntimeError(msg) from exc


INPUT_IMAGES_NAME = "INPUT_IMAGES"
THRESHOLDS_NAME = "THRESHOLDS"
OUTPUT_NAME = "OUTPUT"
TRITON_MODEL_PREFERRED = "yolox_ensemble"
TRITON_MODEL_ALIASES = {TRITON_MODEL_PREFERRED, "yolox"}


class ServiceSettings(BaseSettings):
    model_name: str = Field("nemoretriever-graphic-elements-v1", alias="MODEL_ID")
    model_version: str = Field("1.0.0", alias="YOLOX_TAG")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    threshold: float = Field(0.1, alias="THRESHOLD")
    device: str | None = Field(None, alias="DEVICE")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    auth_token: str | None = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: str | None = Field(None, alias="NIM_NGC_API_KEY")
    nvidia_auth_token: str | None = Field(None, alias="NVIDIA_API_KEY")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    @property
    def resolved_auth_tokens(self) -> set[str]:
        return {token for token in (self.auth_token, self.fallback_auth_token, self.nvidia_auth_token) if token}


settings = ServiceSettings()
logging.basicConfig(level=logging.DEBUG if settings.log_verbose else logging.INFO)
logger = logging.getLogger("nemoretriever-graphic-elements-v1")


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
    loaded: bool = False
    inference_count: int = 0
    last_inference_ns: int = 0


@dataclass
class ThresholdPair:
    conf: float
    iou: float


@dataclass
class SystemSharedMemoryRegion:
    name: str
    key: str
    offset: int
    byte_size: int
    handle: shared_memory.SharedMemory


@dataclass
class CudaSharedMemoryRegion:
    name: str
    raw_handle: bytes
    device_id: int
    byte_size: int
    offset: int
    pointer: Optional[int] = None


model_state = ModelState()
_model_state_lock = threading.Lock()


def _is_model_loaded() -> bool:
    with _model_state_lock:
        return model_state.loaded and _model is not None


def _set_model_loaded(value: bool) -> None:
    with _model_state_lock:
        model_state.loaded = value


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


class SharedMemoryRegistry:
    CUDA_IPC_FLAG = 1
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self._system: Dict[str, SystemSharedMemoryRegion] = {}
        self._cuda: Dict[str, CudaSharedMemoryRegion] = {}
        self._lock = threading.Lock()
        self._cudart: ctypes.CDLL | None = None

    def _get_cudart(self) -> ctypes.CDLL:
        if self._cudart is not None:
            return self._cudart
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
                self._cudart = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        if self._cudart is None:
            raise RuntimeError("CUDA runtime library not found")
        self._cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
        self._cudart.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
        self._cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self._cudart.cudaMemcpy.restype = ctypes.c_int
        self._cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._cudart.cudaSetDevice.restype = ctypes.c_int
        self._cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        return self._cudart

    @staticmethod
    def _validate_positive(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        if name != "offset" and value == 0:
            raise ValueError(f"{name} must be greater than zero")

    def _check_cuda(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUDA error code {result}")

    def _close_system_locked(self, name: str) -> None:
        if name in self._system:
            try:
                self._system[name].handle.close()
            finally:
                del self._system[name]

    def _close_cuda_locked(self, name: str) -> None:
        if name in self._cuda:
            region = self._cuda[name]
            cudart = self._get_cudart()
            try:
                if region.pointer is not None:
                    self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
                    self._check_cuda(
                        cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(region.pointer)), "cudaIpcCloseMemHandle"
                    )
            finally:
                del self._cuda[name]

    def register_system(self, name: str, key: str, byte_size: int, offset: int) -> None:
        self._validate_positive("byte_size", int(byte_size))
        self._validate_positive("offset", int(offset))
        if not name or not key:
            raise ValueError("name and key are required for system shared memory registration")
        try:
            handle = shared_memory.SharedMemory(name=key)
        except FileNotFoundError as exc:
            raise ValueError(f"System shared memory key '{key}' not found") from exc
        except Exception as exc:  # pragma: no cover - surfaced to client
            raise RuntimeError(f"Failed to attach to system shared memory '{key}': {exc}") from exc

        if offset + byte_size > handle.size:
            handle.close()
            raise ValueError("System shared memory segment is smaller than requested byte_size + offset")

        region = SystemSharedMemoryRegion(
            name=name,
            key=key,
            offset=int(offset),
            byte_size=int(byte_size),
            handle=handle,
        )
        with self._lock:
            if name in self._cuda:
                handle.close()
                raise ValueError(f"Region '{name}' is already registered as CUDA shared memory")
            self._close_system_locked(name)
            self._system[name] = region

    def register_cuda(self, name: str, raw_handle: bytes, device_id: int, byte_size: int, offset: int) -> None:
        self._validate_positive("byte_size", int(byte_size))
        self._validate_positive("offset", int(offset))
        if not name or not raw_handle:
            raise ValueError("name and raw_handle are required for CUDA shared memory registration")
        if len(raw_handle) < 1:
            raise ValueError("raw_handle is empty")

        cudart = self._get_cudart()
        self._check_cuda(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")

        handle_buffer = ctypes.create_string_buffer(raw_handle)
        device_pointer = ctypes.c_void_p()
        self._check_cuda(
            cudart.cudaIpcOpenMemHandle(
                ctypes.byref(device_pointer),
                ctypes.cast(handle_buffer, ctypes.c_void_p),
                self.CUDA_IPC_FLAG,
            ),
            "cudaIpcOpenMemHandle",
        )

        region = CudaSharedMemoryRegion(
            name=name,
            raw_handle=bytes(raw_handle),
            device_id=int(device_id),
            byte_size=int(byte_size),
            offset=int(offset),
            pointer=int(device_pointer.value) if device_pointer.value is not None else None,
        )

        with self._lock:
            if name in self._system:
                self._check_cuda(cudart.cudaIpcCloseMemHandle(device_pointer), "cudaIpcCloseMemHandle")
                raise ValueError(f"Region '{name}' is already registered as system shared memory")
            self._close_cuda_locked(name)
            self._cuda[name] = region

    def unregister_system(self, name: str) -> None:
        with self._lock:
            if name not in self._system:
                raise ValueError(f"System shared memory region '{name}' is not registered")
            self._close_system_locked(name)

    def unregister_cuda(self, name: str) -> None:
        with self._lock:
            if name not in self._cuda:
                raise ValueError(f"CUDA shared memory region '{name}' is not registered")
            self._close_cuda_locked(name)

    def system_regions(self) -> List[SystemSharedMemoryRegion]:
        with self._lock:
            return list(self._system.values())

    def cuda_regions(self) -> List[CudaSharedMemoryRegion]:
        with self._lock:
            return list(self._cuda.values())

    def _resolve_region(self, name: str) -> Tuple[str, SystemSharedMemoryRegion | CudaSharedMemoryRegion]:
        with self._lock:
            if name in self._system:
                return "system", self._system[name]
            if name in self._cuda:
                return "cuda", self._cuda[name]
        raise KeyError(f"Shared memory region '{name}' is not registered")

    def _read_system(self, region: SystemSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        start = region.offset + offset
        end = start + byte_size
        if end > region.offset + region.byte_size:
            raise ValueError("Requested read exceeds registered system shared memory bounds")
        return bytes(region.handle.buf[start:end])

    def _write_system(self, region: SystemSharedMemoryRegion, data: bytes, offset: int) -> None:
        start = region.offset + offset
        end = start + len(data)
        if end > region.offset + region.byte_size:
            raise ValueError("Requested write exceeds registered system shared memory bounds")
        region.handle.buf[start:end] = data

    def _read_cuda(self, region: CudaSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        cudart = self._get_cudart()
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        start = region.offset + offset
        end = start + byte_size
        if end > region.byte_size:
            raise ValueError("Requested read exceeds registered CUDA shared memory bounds")
        host_buffer = (ctypes.c_char * byte_size)()
        device_ptr = ctypes.c_void_p(region.pointer + start)
        self._check_cuda(
            cudart.cudaMemcpy(
                ctypes.cast(host_buffer, ctypes.c_void_p),
                device_ptr,
                ctypes.c_size_t(byte_size),
                self.CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy (device to host)",
        )
        return bytes(host_buffer)

    def _write_cuda(self, region: CudaSharedMemoryRegion, data: bytes, offset: int) -> None:
        cudart = self._get_cudart()
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        start = region.offset + offset
        end = start + len(data)
        if end > region.byte_size:
            raise ValueError("Requested write exceeds registered CUDA shared memory bounds")
        host_buffer = ctypes.create_string_buffer(data)
        device_ptr = ctypes.c_void_p(region.pointer + start)
        self._check_cuda(
            cudart.cudaMemcpy(
                device_ptr,
                ctypes.cast(host_buffer, ctypes.c_void_p),
                ctypes.c_size_t(len(data)),
                self.CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy (host to device)",
        )

    def read(self, name: str, byte_size: int, offset: int) -> bytes:
        if byte_size <= 0:
            raise ValueError("Requested byte_size must be greater than zero")
        kind, region = self._resolve_region(name)
        if kind == "system":
            return self._read_system(region, byte_size, offset)
        return self._read_cuda(region, byte_size, offset)

    def write(self, name: str, data: bytes, offset: int) -> None:
        if not data:
            return
        kind, region = self._resolve_region(name)
        if kind == "system":
            self._write_system(region, data, offset)
            return
        self._write_cuda(region, data, offset)


shared_memory_registry = SharedMemoryRegistry()

REQUEST_COUNTER = Counter("nim_requests_total", "Total requests served", ["protocol", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
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
INFLIGHT_GAUGE = Gauge("nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"])

async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
sync_rate_limiter = SyncRateLimiter(settings.rate_limit)

_metrics_started = False
if settings.metrics_port and not settings.require_auth and not _metrics_started:
    start_http_server(settings.metrics_port)
    _metrics_started = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


class InferInput(BaseModel):
    type: Literal["image_url"]
    url: str | HttpUrl


class InferRequest(BaseModel):
    input: List[InferInput]

    @field_validator("input")
    @classmethod
    def ensure_not_empty(cls, value: List[InferInput]) -> List[InferInput]:
        if not value:
            raise ValueError("input must contain at least one image")
        return value


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


class InferResponseItem(BaseModel):
    bounding_boxes: Dict[str, List[BoundingBox]]


DEFAULT_LABELS: List[str] = [
    "chart_title",
    "x_title",
    "y_title",
    "xlabel",
    "ylabel",
    "other",
    "legend_label",
    "legend_title",
    "mark_label",
    "value_label",
]
DEFAULT_CONF_THRESHOLD = 0.01
DEFAULT_IOU_THRESHOLD = 0.25

torch.set_grad_enabled(False)
_model_lock = threading.Lock()
_model: torch.nn.Module | None = None
MODEL_LABELS: List[str] = list(DEFAULT_LABELS)
_base_conf_threshold: float = DEFAULT_CONF_THRESHOLD
_base_iou_threshold: float = DEFAULT_IOU_THRESHOLD


def _instantiate_model() -> torch.nn.Module:
    model = define_model("graphic_element_v1")
    if settings.device:
        model = model.to(settings.device)
    model.threshold = settings.threshold
    model.eval()
    return model


def _load_model() -> None:
    global _model, MODEL_LABELS, _base_conf_threshold, _base_iou_threshold
    created = False
    with _model_lock:
        if _model is None:
            _model = _instantiate_model()
            MODEL_LABELS = list(getattr(_model, "labels", MODEL_LABELS))
            _base_conf_threshold = float(getattr(_model, "conf_thresh", _base_conf_threshold))
            _base_iou_threshold = float(getattr(_model, "iou_thresh", _base_iou_threshold))
            created = True
    _set_model_loaded(True)
    if created:
        logger.info(
            "Model loaded (version=%s, conf_thresh=%.4f, iou_thresh=%.4f)",
            settings.model_version,
            _base_conf_threshold,
            _base_iou_threshold,
        )
    else:
        logger.info("Model marked as loaded")


def _unload_model() -> None:
    global _model
    with _model_lock:
        _model = None
    _set_model_loaded(False)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("CUDA cache clear failed during unload", exc_info=True)


_load_model()
app = FastAPI(title=settings.model_name, version=settings.model_version)


def _extract_token(header_value: Optional[str]) -> Optional[str]:
    if header_value is None:
        return None
    stripped = header_value.strip()
    if stripped == "":
        return None
    parts = stripped.split(" ", maxsplit=1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token if token else None
    return stripped


def _is_authorized(header_value: Optional[str]) -> bool:
    token = _extract_token(header_value)
    if not settings.require_auth:
        return True
    if token is None:
        return False
    if settings.resolved_auth_tokens:
        return token in settings.resolved_auth_tokens
    return True


async def _require_http_auth(request: Request) -> None:
    header = request.headers.get("authorization") or request.headers.get("ngc-api-key")
    if _is_authorized(header):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def _require_grpc_auth(context: grpc.ServicerContext) -> None:
    metadata: Mapping[str, str] = {k.lower(): v for k, v in context.invocation_metadata()}
    header = metadata.get("authorization") or metadata.get("ngc-api-key")
    if _is_authorized(header):
        return
    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")


def _decode_data_url(data_url: str) -> bytes:
    try:
        _, b64_payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise ValueError("Expected data URL in the form data:image/<type>;base64,<payload>.") from exc
    return base64.b64decode(b64_payload)


def _decode_base64_image(payload: str) -> np.ndarray:
    encoded = payload
    if "base64," in payload:
        _, encoded = payload.split("base64,", maxsplit=1)
    try:
        image_bytes = base64.b64decode(encoded)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def _load_image(url: str) -> np.ndarray:
    if url.startswith("data:"):
        payload = _decode_data_url(url)
        return np.array(Image.open(io.BytesIO(payload)).convert("RGB"))

    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=settings.request_timeout_seconds)
        response.raise_for_status()
        return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))

    return np.array(Image.open(url).convert("RGB"))


def _postprocess_prediction(
    prediction: Dict[str, np.ndarray], score_threshold: float
) -> Dict[str, List[List[float]]]:
    boxes, labels, scores = postprocess_preds_graphic_element(
        prediction, threshold=score_threshold, class_labels=MODEL_LABELS
    )
    bbox_dict: Dict[str, List[List[float]]] = {label: [] for label in MODEL_LABELS}
    for bbox, label_idx, score in zip(boxes, labels, scores):
        label_name = MODEL_LABELS[int(label_idx)]
        bbox_dict[label_name].append(
            [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(score),
            ]
        )
    return bbox_dict


def _predict_raw(
    images: List[np.ndarray], threshold_pairs: List[ThresholdPair], score_threshold: float
) -> List[Dict[str, List[List[float]]]]:
    if not images:
        return []
    if _model is None:
        raise RuntimeError("Model is not loaded")
    if len(threshold_pairs) != len(images):
        raise ValueError("THRESHOLDS count must match number of images")

    tensors = [_model.preprocess(image) for image in images]
    results: List[Dict[str, List[List[float]]]] = []
    unique_pairs = {(pair.conf, pair.iou) for pair in threshold_pairs}

    try:
        if len(unique_pairs) == 1:
            pair = threshold_pairs[0]
            _model.conf_thresh = pair.conf
            _model.iou_thresh = pair.iou
            batch = torch.stack(tensors)
            sizes: List[Tuple[int, int]] = [(image.shape[0], image.shape[1]) for image in images]
            with torch.inference_mode():
                preds = _model(batch, sizes)
            results.extend([_postprocess_prediction(pred, score_threshold) for pred in preds])
            return results

        for tensor, image, pair in zip(tensors, images, threshold_pairs):
            _model.conf_thresh = pair.conf
            _model.iou_thresh = pair.iou
            with torch.inference_mode():
                pred = _model(torch.stack([tensor]), [(image.shape[0], image.shape[1])])[0]
            results.append(_postprocess_prediction(pred, score_threshold))
    finally:
        _model.conf_thresh = _base_conf_threshold
        _model.iou_thresh = _base_iou_threshold

    return results


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
            for label in MODEL_LABELS
        }
    }


def _serialize_predictions(raw_predictions: List[Dict[str, List[List[float]]]]) -> bytes:
    payload = [json.dumps(pred) for pred in raw_predictions]
    tensor = np.array([p.encode("utf-8") for p in payload], dtype=np.object_)
    return serialize_byte_tensor(tensor)


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


def _extract_shared_memory_params(
    parameters: Mapping[str, service_pb2.InferParameter],
) -> Optional[Tuple[str, int, int]]:
    region_param = parameters.get("shared_memory_region")
    if region_param is None or region_param.WhichOneof("parameter_choice") != "string_param":
        return None
    region_name = region_param.string_param
    if not region_name:
        return None

    size_param = parameters.get("shared_memory_byte_size")
    if size_param is None or not size_param.WhichOneof("parameter_choice"):
        raise ValueError("shared_memory_byte_size is required for shared memory tensors")
    byte_size = int(size_param.int64_param or size_param.uint64_param)
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be greater than zero")

    offset_param = parameters.get("shared_memory_offset")
    offset = 0
    if offset_param and offset_param.WhichOneof("parameter_choice"):
        offset = int(offset_param.int64_param or offset_param.uint64_param)
        if offset < 0:
            raise ValueError("shared_memory_offset must be non-negative")

    return region_name, byte_size, offset


def _prepare_threshold_pairs(threshold_array: Optional[np.ndarray], batch_size: int) -> List[ThresholdPair]:
    if threshold_array is None or threshold_array.size == 0:
        return [ThresholdPair(conf=_base_conf_threshold, iou=_base_iou_threshold) for _ in range(batch_size)]

    thresholds = threshold_array
    if thresholds.ndim == 1:
        if thresholds.size % 2 != 0:
            raise ValueError("THRESHOLDS must provide pairs of [conf_threshold, iou_threshold]")
        thresholds = thresholds.reshape((-1, 2))
    if thresholds.shape[1] < 2:
        raise ValueError("THRESHOLDS must have a trailing dimension of 2")

    rows = thresholds.shape[0]
    if rows not in (1, batch_size):
        raise ValueError(f"THRESHOLDS batch dimension must be 1 or {batch_size}")

    pairs: List[ThresholdPair] = []
    for idx in range(batch_size):
        conf, iou = thresholds[0 if rows == 1 else idx][:2]
        if not np.isfinite(conf) or not np.isfinite(iou):
            raise ValueError("THRESHOLDS values must be finite")
        if conf < 0 or iou < 0:
            raise ValueError("THRESHOLDS values must be non-negative")
        pairs.append(ThresholdPair(conf=float(conf), iou=float(iou)))
    return pairs


def _record_metrics(endpoint: str, protocol: str, status: str, start_time: float) -> None:
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(time.perf_counter() - start_time)


def _record_inference_stats() -> None:
    with _model_state_lock:
        model_state.inference_count += 1
        model_state.last_inference_ns = time.time_ns()


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
    return {"ready": _is_model_loaded()}


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


@app.post("/v1/infer")
async def infer(
    request_body: InferRequest, _: None = Depends(_require_http_auth)
) -> Dict[str, List[Dict[str, Dict[str, List[Dict[str, float]]]]]]:
    start_time = time.perf_counter()
    endpoint = "/v1/infer"
    if not _is_model_loaded():
        _record_metrics(endpoint, "http", "not_loaded", start_time)
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if len(request_body.input) > settings.max_batch_size:
        _record_metrics(endpoint, "http", "batch_limit", start_time)
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request_body.input)} exceeds limit {settings.max_batch_size}",
        )

    try:
        images = [_load_image(item.url) for item in request_body.input]
    except (OSError, ValueError, ValidationError) as exc:
        logger.exception("Invalid image payload.")
        _record_metrics(endpoint, "http", "invalid_payload", start_time)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        logger.exception("Failed to fetch remote image.")
        _record_metrics(endpoint, "http", "invalid_payload", start_time)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).inc()
        tracer_ctx = TRACER.start_as_current_span("http_infer") if TRACER else nullcontext()
        with tracer_ctx:
            async with async_rate_limiter.limit():
                threshold_pairs = _prepare_threshold_pairs(None, len(images))
                with _model_lock:
                    raw_predictions = _predict_raw(images, threshold_pairs, settings.threshold)
    except RuntimeError as exc:  # pragma: no cover - surfaced to client
        logger.exception("Model unavailable.")
        _record_metrics(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        logger.exception("Model inference failed.")
        _record_metrics(endpoint, "http", "error", start_time)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).dec()

    predictions = [_format_http_prediction(pred) for pred in raw_predictions]
    _record_metrics(endpoint, "http", "success", start_time)
    return {"data": predictions}


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "nemoretriever-graphic-elements-v1 HTTP"}, status_code=200)


def _normalize_model_name(model_name: Optional[str]) -> str:
    if model_name is None or model_name == "":
        return TRITON_MODEL_PREFERRED
    if model_name in TRITON_MODEL_ALIASES:
        return model_name
    raise ValueError(f"Unknown model {model_name}")


def _model_ready_response() -> service_pb2.ModelReadyResponse:
    return service_pb2.ModelReadyResponse(ready=_is_model_loaded())


def _server_ready_response() -> service_pb2.ServerReadyResponse:
    return service_pb2.ServerReadyResponse(ready=_is_model_loaded())


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
    loaded = _is_model_loaded()
    models = [
        service_pb2.RepositoryIndexResponse.ModelIndex(
            name=model_name,
            version=settings.model_version,
            state="READY" if loaded else "UNAVAILABLE",
            reason="" if loaded else "Model is not loaded",
        )
        for model_name in sorted(TRITON_MODEL_ALIASES)
    ]
    return service_pb2.RepositoryIndexResponse(models=models)


@app.get("/v2/models/{model_name}")
async def model_metadata_http(model_name: str, _: None = Depends(_require_http_auth)) -> JSONResponse:
    try:
        normalized = _normalize_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response = _model_metadata_response(normalized)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/config")
async def model_config_http(model_name: str, _: None = Depends(_require_http_auth)) -> JSONResponse:
    try:
        normalized = _normalize_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    response = _model_config_response(normalized)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> JSONResponse:
    try:
        _ = _normalize_model_name(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse({"ready": _is_model_loaded()})


class TritonService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def ServerLive(self, request, context):
        _require_grpc_auth(context)
        return _server_live_response()

    def ServerReady(self, request, context):
        _require_grpc_auth(context)
        return _server_ready_response()

    def ModelReady(self, request, context):
        _require_grpc_auth(context)
        try:
            _ = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return _model_ready_response()

    def ServerMetadata(self, request, context):
        _require_grpc_auth(context)
        return service_pb2.ServerMetadataResponse(
            name=settings.model_name,
            version=settings.model_version,
            extensions=["model_repository", "metrics"],
        )

    def ModelMetadata(self, request, context):
        _require_grpc_auth(context)
        try:
            model_name = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return _model_metadata_response(model_name)

    def ModelConfig(self, request, context):
        _require_grpc_auth(context)
        try:
            model_name = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return _model_config_response(model_name)

    def RepositoryIndex(self, request, context):
        _require_grpc_auth(context)
        return _model_repository_index()

    def RepositoryModelLoad(self, request, context):
        _require_grpc_auth(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            _ = _normalize_model_name(request.model_name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        _load_model()
        return service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(self, request, context):
        _require_grpc_auth(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            _ = _normalize_model_name(request.model_name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        _unload_model()
        return service_pb2.RepositoryModelUnloadResponse()

    def ModelInfer(self, request, context):
        start_time = time.perf_counter()
        endpoint = "grpc::infer"
        metrics_recorded = False
        _require_grpc_auth(context)
        try:
            model_name = _normalize_model_name(request.model_name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
            metrics_recorded = True
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        if not _is_model_loaded():
            _record_metrics(endpoint, "grpc", "not_loaded", start_time)
            metrics_recorded = True
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        if _model is None:
            _record_metrics(endpoint, "grpc", "not_loaded", start_time)
            metrics_recorded = True
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is unavailable")
        try:
            input_lookup: Dict[str, int] = {inp.name: idx for idx, inp in enumerate(request.inputs)}
            if INPUT_IMAGES_NAME not in input_lookup:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                metrics_recorded = True
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INPUT_IMAGES missing")
            image_index = input_lookup[INPUT_IMAGES_NAME]
            thresholds_index = input_lookup.get(THRESHOLDS_NAME)

            image_tensor = request.inputs[image_index]
            try:
                image_shm = _extract_shared_memory_params(image_tensor.parameters)
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                metrics_recorded = True
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            image_raw = None
            if image_shm:
                try:
                    image_raw = shared_memory_registry.read(*image_shm)
                except Exception as exc:  # pragma: no cover - surfaced to client
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    metrics_recorded = True
                    status = grpc.StatusCode.NOT_FOUND if isinstance(exc, KeyError) else grpc.StatusCode.FAILED_PRECONDITION
                    context.abort(status, f"Failed to read shared memory '{image_shm[0]}': {exc}")
            elif image_index < len(request.raw_input_contents):
                image_raw = request.raw_input_contents[image_index]
            image_strings = _deserialize_bytes_input(image_tensor, image_raw)

            threshold_array = None
            if thresholds_index is not None:
                threshold_tensor = request.inputs[thresholds_index]
                try:
                    threshold_shm = _extract_shared_memory_params(threshold_tensor.parameters)
                except ValueError as exc:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    metrics_recorded = True
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

                threshold_raw = None
                if threshold_shm:
                    try:
                        threshold_raw = shared_memory_registry.read(*threshold_shm)
                    except Exception as exc:  # pragma: no cover - surfaced to client
                        _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                        metrics_recorded = True
                        status = grpc.StatusCode.NOT_FOUND if isinstance(exc, KeyError) else grpc.StatusCode.FAILED_PRECONDITION
                        context.abort(
                            status,
                            f"Failed to read shared memory '{threshold_shm[0]}': {exc}",
                        )
                elif thresholds_index < len(request.raw_input_contents):
                    threshold_raw = request.raw_input_contents[thresholds_index]
                threshold_array = _deserialize_fp32_input(threshold_tensor, threshold_raw)

            flat_images = [s for s in image_strings if s]
            batch_size = len(flat_images)
            if batch_size == 0:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                metrics_recorded = True
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No images provided")
            if batch_size > settings.max_batch_size:
                _record_metrics(endpoint, "grpc", "batch_limit", start_time)
                metrics_recorded = True
                context.abort(
                    grpc.StatusCode.OUT_OF_RANGE,
                    f"Batch size {batch_size} exceeds limit {settings.max_batch_size}",
                )

            try:
                threshold_pairs = _prepare_threshold_pairs(threshold_array, batch_size)
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                metrics_recorded = True
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            try:
                images: List[np.ndarray] = [_decode_base64_image(value) for value in flat_images]
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                metrics_recorded = True
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).inc()
            try:
                tracer_ctx = TRACER.start_as_current_span("grpc_infer") if TRACER else nullcontext()
                with tracer_ctx:
                    with sync_rate_limiter.limit():
                        with _model_lock:
                            raw_predictions = _predict_raw(images, threshold_pairs, settings.threshold)
            finally:
                INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).dec()

            serialized = _serialize_predictions(raw_predictions)
            output = service_pb2.ModelInferResponse.InferOutputTensor(
                name=OUTPUT_NAME, datatype="BYTES", shape=[batch_size]
            )
            response = service_pb2.ModelInferResponse(model_name=model_name, outputs=[output], id=request.id)
            response.model_version = settings.model_version
            output_lookup: Dict[str, service_pb2.ModelInferRequest.InferRequestedOutputTensor] = {
                out.name: out for out in request.outputs
            }
            output_shm = None
            if OUTPUT_NAME in output_lookup:
                try:
                    output_shm = _extract_shared_memory_params(output_lookup[OUTPUT_NAME].parameters)
                except ValueError as exc:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    metrics_recorded = True
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            if output_shm:
                region, byte_size, offset = output_shm
                if len(serialized) > byte_size:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    metrics_recorded = True
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Shared memory region '{region}' is too small for output",
                    )
                try:
                    shared_memory_registry.write(region, serialized, offset)
                except KeyError as exc:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    metrics_recorded = True
                    context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
                except Exception as exc:  # pragma: no cover - surfaced to client
                    _record_metrics(endpoint, "grpc", "error", start_time)
                    metrics_recorded = True
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"Failed to write output to shared memory: {exc}")
            else:
                response.raw_output_contents.append(serialized)
            _record_metrics(endpoint, "grpc", "success", start_time)
            metrics_recorded = True
            _record_inference_stats()
            return response
        except grpc.RpcError:
            if not metrics_recorded:
                _record_metrics(endpoint, "grpc", "error", start_time)
            raise
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("gRPC inference failed: %s", exc)
            if not metrics_recorded:
                _record_metrics(endpoint, "grpc", "error", start_time)
            context.abort(grpc.StatusCode.INTERNAL, "Inference failed")

    def ModelStatistics(self, request, context):
        _require_grpc_auth(context)
        try:
            _ = _normalize_model_name(request.name or TRITON_MODEL_PREFERRED)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        stats = service_pb2.ModelStatistics(
            name=settings.model_name,
            version=settings.model_version,
            last_inference=model_state.last_inference_ns or 0,
            inference_count=model_state.inference_count,
            execution_count=model_state.inference_count,
        )
        stats.inference_stats.success.count = model_state.inference_count
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    def ModelStreamInfer(self, request_iterator, context):
        _require_grpc_auth(context)
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Streaming is not supported")

    def SystemSharedMemoryStatus(self, request, context):
        _require_grpc_auth(context)
        response = service_pb2.SystemSharedMemoryStatusResponse()
        regions = shared_memory_registry.system_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"System shared memory region '{request.name}' not found")
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.key = region.key
            status.offset = region.offset
            status.byte_size = region.byte_size
        return response

    def SystemSharedMemoryRegister(self, request, context):
        _require_grpc_auth(context)
        try:
            shared_memory_registry.register_system(request.name, request.key, request.byte_size, request.offset)
            return service_pb2.SystemSharedMemoryRegisterResponse()
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:  # pragma: no cover - surfaced to client
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def SystemSharedMemoryUnregister(self, request, context):
        _require_grpc_auth(context)
        try:
            shared_memory_registry.unregister_system(request.name)
            return service_pb2.SystemSharedMemoryUnregisterResponse()
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))

    def CudaSharedMemoryStatus(self, request, context):
        _require_grpc_auth(context)
        response = service_pb2.CudaSharedMemoryStatusResponse()
        regions = shared_memory_registry.cuda_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' not found")
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.device_id = region.device_id
            status.byte_size = region.byte_size
        return response

    def CudaSharedMemoryRegister(self, request, context):
        _require_grpc_auth(context)
        try:
            shared_memory_registry.register_cuda(
                request.name, request.raw_handle, request.device_id, request.byte_size, 0
            )
            return service_pb2.CudaSharedMemoryRegisterResponse()
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:  # pragma: no cover - surfaced to client
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def CudaSharedMemoryUnregister(self, request, context):
        _require_grpc_auth(context)
        try:
            shared_memory_registry.unregister_cuda(request.name)
            return service_pb2.CudaSharedMemoryUnregisterResponse()
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))


def _start_grpc_server() -> grpc.Server:
    grpc_workers = max(settings.rate_limit or 4, 4)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=grpc_workers))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(TritonService(), server)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
