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
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import grpc
import numpy as np
import requests
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from google.protobuf import json_format
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor

try:
    from nemotron_ocr.inference.pipeline import NemotronOCR
except ImportError as exc:  # pragma: no cover - configuration error surfaced to operator
    msg = (
        "nemotron_ocr is missing. Install the HF repo (git lfs install && git clone "
        "https://huggingface.co/nvidia/nemotron-ocr-v1 && pip install -v ./nemotron-ocr-v1/nemotron-ocr)."
    )
    raise RuntimeError(msg) from exc


MergeLevel = Literal["word", "sentence", "paragraph"]

DEFAULT_MODEL_NAME = "scene_text_wrapper"
DEFAULT_MODEL_VERSION = "1.1.0"


class Settings(BaseSettings):
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_TRITON_GRPC_PORT")
    metrics_port: int = Field(8002, alias="NIM_TRITON_METRICS_PORT")
    merge_level: MergeLevel = Field("paragraph", alias="MERGE_LEVEL")
    max_batch_size: int = Field(8, alias="NIM_TRITON_MAX_BATCH_SIZE")
    model_dir: str = Field("./checkpoints", alias="MODEL_DIR")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    model_version: str = Field(DEFAULT_MODEL_VERSION, alias="MODEL_VERSION")
    model_name: str = Field(DEFAULT_MODEL_NAME, alias="OCR_MODEL_NAME")
    auth_token: Optional[str] = Field(None, alias="NGC_API_KEY")
    alt_auth_token: Optional[str] = Field(None, alias="NVIDIA_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    otel_endpoint: Optional[str] = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class InferInput(BaseModel):
    type: Literal["image_url"]
    url: str | HttpUrl


class InferRequest(BaseModel):
    input: List[InferInput]
    merge_levels: List[MergeLevel] | None = None


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    points: List[Point]
    type: Literal["quadrilateral"] = "quadrilateral"


class TextPrediction(BaseModel):
    text: str
    confidence: float


class TextDetection(BaseModel):
    bounding_box: BoundingBox
    text_prediction: TextPrediction


class InferResponseItem(BaseModel):
    text_detections: List[TextDetection]


settings = Settings()
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("nemoretriever-ocr-v1")

torch.set_grad_enabled(False)


@dataclass
class ModelState:
    loaded: bool = False
    inference_count: int = 0
    success_ns_total: int = 0
    last_inference_ns: int = 0


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

REQUEST_COUNTER = Counter("nim_ocr_requests_total", "Total OCR inference requests", ["protocol"])
REQUEST_LATENCY = Histogram("nim_ocr_request_latency_seconds", "Latency for OCR inference requests", ["protocol"])
BATCH_SIZE = Histogram("nim_ocr_batch_size", "Observed OCR batch sizes", ["protocol"])
_METRICS_STARTED = False

_model_state = ModelState()
_model_state_lock = threading.Lock()
_model_lock = threading.Lock()
_model_instance: Optional[NemotronOCR] = None

grpc_server: Optional[grpc.aio.Server] = None
_triton_service: Optional["TritonOCRService"] = None
app = FastAPI(title="nemoretriever-ocr-v1", version=settings.model_version)


def _configure_tracer() -> trace.Tracer:
    if not settings.otel_endpoint:
        return trace.get_tracer("nemoretriever-ocr-v1")

    resource = Resource.create({"service.name": "nemoretriever-ocr-v1"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer("nemoretriever-ocr-v1")


tracer = _configure_tracer()


class SharedMemoryRegistry:
    CUDA_IPC_FLAG = 1
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self._system: Dict[str, SystemSharedMemoryRegion] = {}
        self._cuda: Dict[str, CudaSharedMemoryRegion] = {}
        self._lock = threading.Lock()
        self._cudart: Optional[ctypes.CDLL] = None

    def _validate_positive(self, field: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{field} must be greater than zero")

    def _validate_non_negative(self, field: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{field} must be non-negative")

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

    def _check_cuda(self, status: int, op: str) -> None:
        if status != 0:
            raise RuntimeError(f"{op} failed with status {status}")

    def _close_system_locked(self, name: str) -> None:
        region = self._system.pop(name, None)
        if region is None:
            return
        try:
            region.handle.close()
        except Exception:
            logger.debug("Failed to close shared memory handle for %s", name, exc_info=True)

    def _close_cuda_locked(self, name: str) -> None:
        region = self._cuda.pop(name, None)
        if region is None or region.pointer is None:
            return
        try:
            cudart = self._get_cudart()
            self._check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(region.pointer)), "cudaIpcCloseMemHandle")
        except Exception:
            logger.debug("Failed to close CUDA shared memory handle for %s", name, exc_info=True)

    def register_system(self, name: str, key: str, byte_size: int, offset: int) -> None:
        self._validate_positive("byte_size", int(byte_size))
        self._validate_non_negative("offset", int(offset))
        if not name or not key:
            raise ValueError("name and key are required for system shared memory registration")
        handle = shared_memory.SharedMemory(name=key, create=False)
        region = SystemSharedMemoryRegion(name=name, key=key, offset=int(offset), byte_size=int(byte_size), handle=handle)
        with self._lock:
            if name in self._cuda:
                handle.close()
                raise ValueError(f"Region '{name}' is already registered as CUDA shared memory")
            if name in self._system:
                handle.close()
                raise ValueError(f"System shared memory region '{name}' is already registered")
            self._system[name] = region

    def unregister_system(self, name: str) -> None:
        with self._lock:
            if name not in self._system:
                raise ValueError(f"System shared memory region '{name}' is not registered")
            self._close_system_locked(name)

    def register_cuda(self, name: str, raw_handle: bytes, device_id: int, byte_size: int, offset: int) -> None:
        self._validate_positive("byte_size", int(byte_size))
        self._validate_non_negative("offset", int(offset))
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
            raise KeyError(name)

    def _read_system(self, region: SystemSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        start = region.offset + offset
        end = start + byte_size
        if start < region.offset or end > region.offset + region.byte_size:
            raise ValueError("System shared memory read out of bounds")
        return bytes(region.handle.buf[start:end])

    def _read_cuda(self, region: CudaSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        cudart = self._get_cudart()
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        start = region.offset + offset
        end = start + byte_size
        if end > region.byte_size:
            raise ValueError("Requested read exceeds registered CUDA shared memory bounds")
        host_buffer = ctypes.create_string_buffer(byte_size)
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
        return bytes(host_buffer.raw)

    def _write_system(self, region: SystemSharedMemoryRegion, data: bytes, offset: int) -> None:
        start = region.offset + offset
        end = start + len(data)
        if start < region.offset or end > region.offset + region.byte_size:
            raise ValueError("System shared memory write out of bounds")
        region.handle.buf[start:end] = data

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
        self._validate_positive("byte_size", int(byte_size))
        self._validate_non_negative("offset", int(offset))
        kind, region = self._resolve_region(name)
        if kind == "system":
            return self._read_system(region, int(byte_size), int(offset))
        return self._read_cuda(region, int(byte_size), int(offset))

    def write(self, name: str, data: bytes, offset: int) -> None:
        self._validate_non_negative("offset", int(offset))
        if not data:
            return
        kind, region = self._resolve_region(name)
        if kind == "system":
            self._write_system(region, data, int(offset))
            return
        self._write_cuda(region, data, int(offset))


shared_memory_registry = SharedMemoryRegistry()


def _set_model_loaded(value: bool) -> None:
    with _model_state_lock:
        _model_state.loaded = value


def _is_model_loaded() -> bool:
    with _model_state_lock:
        return _model_state.loaded and _model_instance is not None


def _load_model() -> None:
    global _model_instance
    with _model_lock:
        if _model_instance is not None:
            _set_model_loaded(True)
            return
        try:
            _model_instance = NemotronOCR(model_dir=settings.model_dir)
        except Exception:
            _set_model_loaded(False)
            raise
    _set_model_loaded(True)
    logger.info("Model loaded from %s", settings.model_dir)


def _unload_model() -> None:
    global _model_instance
    with _model_lock:
        _model_instance = None
    _set_model_loaded(False)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("CUDA cache clear failed during unload", exc_info=True)


def _get_model() -> NemotronOCR:
    if not _is_model_loaded():
        raise RuntimeError("Model is not loaded")
    model = _model_instance
    if model is None:
        raise RuntimeError("Model is not loaded")
    return model


def _record_inference_success(duration_ns: int) -> None:
    with _model_state_lock:
        _model_state.inference_count += 1
        _model_state.success_ns_total += duration_ns
        _model_state.last_inference_ns = time.time_ns()


def _current_model_ready() -> bool:
    return _is_model_loaded()


@app.middleware("http")
async def _http_tracing(request: Request, call_next):
    span_name = f"http {request.method.lower()} {request.url.path}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            span.record_exception(exc)
            span.set_status(Status(status_code=StatusCode.ERROR))
            raise
        span.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 400:
            span.set_status(Status(status_code=StatusCode.ERROR))
        return response


def _start_metrics_server() -> None:
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    if _expected_token():
        _METRICS_STARTED = True
        logger.info("Metrics server requires auth; use /metrics endpoints instead of the Prometheus port.")
        return
    start_http_server(settings.metrics_port)
    _METRICS_STARTED = True
    logger.info("Started metrics server on :%d/metrics", settings.metrics_port)


def _decode_data_url(data_url: str) -> bytes:
    try:
        _, payload = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise ValueError("Expected data URL in the form data:image/<type>;base64,<payload>.") from exc
    return base64.b64decode(payload)


def _load_image(url: str) -> np.ndarray:
    if url.startswith("data:"):
        return np.array(Image.open(io.BytesIO(_decode_data_url(url))).convert("RGB"))

    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=settings.request_timeout_seconds)
        response.raise_for_status()
        return np.array(Image.open(io.BytesIO(response.content)).convert("RGB"))

    return np.array(Image.open(url).convert("RGB"))


def _load_base64_image(encoded: str) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    try:
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError("Failed to decode base64 image.") from exc


def _normalize_coord(value: float, dimension: float) -> float:
    normalized = float(value)
    if dimension > 0 and abs(normalized) > 1.0:
        normalized /= dimension
    if dimension > 0:
        normalized = max(0.0, min(1.0, normalized))
    return normalized


def _parse_prediction(pred: dict, width: int, height: int) -> Optional[Tuple[List[List[float]], str, float]]:
    if isinstance(pred.get("left"), str):
        return None

    try:
        left = float(pred.get("left", 0.0))
        right = float(pred.get("right", 0.0))
        upper = float(pred.get("upper", 0.0))
        lower = float(pred.get("lower", 0.0))
    except (TypeError, ValueError):
        return None

    x_min = _normalize_coord(min(left, right), float(width))
    x_max = _normalize_coord(max(left, right), float(width))
    y_min = _normalize_coord(min(lower, upper), float(height))
    y_max = _normalize_coord(max(lower, upper), float(height))

    points = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]

    text = str(pred.get("text", ""))
    try:
        confidence = float(pred.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return points, text, confidence


def _format_predictions(predictions: Sequence[dict], width: int, height: int) -> InferResponseItem:
    detections: List[TextDetection] = []
    for parsed in (_parse_prediction(pred, width, height) for pred in predictions):
        if parsed is None:
            continue
        points, text, confidence = parsed
        bbox_points = [Point(x=pt[0], y=pt[1]) for pt in points]
        detections.append(
            TextDetection(
                bounding_box=BoundingBox(points=bbox_points),
                text_prediction=TextPrediction(text=text, confidence=confidence),
            )
        )
    return InferResponseItem(text_detections=detections)


def _collect_predictions(predictions: Sequence[dict], width: int, height: int) -> Tuple[List[List[List[float]]], List[str], List[float]]:
    boxes: List[List[List[float]]] = []
    texts: List[str] = []
    confidences: List[float] = []
    for parsed in (_parse_prediction(pred, width, height) for pred in predictions):
        if parsed is None:
            continue
        points, text, confidence = parsed
        boxes.append(points)
        texts.append(text)
        confidences.append(confidence)
    return boxes, texts, confidences


def _expected_token() -> Optional[str]:
    return settings.auth_token or settings.alt_auth_token


def _validate_bearer(value: Optional[str], on_error) -> None:
    token = _expected_token()
    if not token:
        return
    if not value:
        on_error("Missing bearer token.")
        return
    if not value.lower().startswith("bearer "):
        on_error("Invalid authorization scheme.")
        return
    provided = value.split(" ", maxsplit=1)[1].strip()
    if provided != token:
        on_error("Invalid bearer token.")


async def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    def _raise(detail: str) -> None:
        raise HTTPException(status_code=401, detail=detail)

    _validate_bearer(authorization, _raise)


def _authorize_grpc(context: grpc.ServicerContext) -> None:
    def _abort(detail: str) -> None:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, detail)

    metadata = {md.key: md.value for md in context.invocation_metadata()}
    auth_value = metadata.get("authorization") or metadata.get("Authorization")
    _validate_bearer(auth_value, _abort)


def _model_config_proto() -> model_config_pb2.ModelConfig:
    cfg = model_config_pb2.ModelConfig(
        name=settings.model_name,
        platform="python",
        max_batch_size=settings.max_batch_size,
    )
    cfg.input.extend(
        [
            model_config_pb2.ModelInput(name="INPUT_IMAGE_URLS", data_type=model_config_pb2.TYPE_BYTES, dims=[1]),
            model_config_pb2.ModelInput(name="MERGE_LEVELS", data_type=model_config_pb2.TYPE_BYTES, dims=[1]),
        ]
    )
    cfg.output.extend(
        [
            model_config_pb2.ModelOutput(name="OUTPUT", data_type=model_config_pb2.TYPE_BYTES, dims=[3]),
        ]
    )
    return cfg


def _model_metadata_response() -> service_pb2.ModelMetadataResponse:
    return service_pb2.ModelMetadataResponse(
        name=settings.model_name,
        versions=[settings.model_version],
        platform="python",
        inputs=[
            service_pb2.ModelMetadataResponse.TensorMetadata(name="INPUT_IMAGE_URLS", datatype="BYTES", shape=[1]),
            service_pb2.ModelMetadataResponse.TensorMetadata(name="MERGE_LEVELS", datatype="BYTES", shape=[1]),
        ],
        outputs=[
            service_pb2.ModelMetadataResponse.TensorMetadata(name="OUTPUT", datatype="BYTES", shape=[3]),
        ],
    )


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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


def _deserialize_input_tensor(raw: Optional[bytes], shape: Sequence[int], name: str) -> np.ndarray:
    if raw is None:
        raise ValueError(f"No data provided for input {name}.")
    flat = deserialize_bytes_tensor(raw)
    if shape:
        try:
            flat = flat.reshape(shape)
        except ValueError as exc:
            raise ValueError(f"Invalid shape for input {name}: {exc}") from exc
    return flat


def _extract_grpc_inputs(request: service_pb2.ModelInferRequest) -> Tuple[List[np.ndarray], List[MergeLevel]]:
    inputs: Dict[str, np.ndarray] = {}
    raw_inputs = list(request.raw_input_contents)
    raw_index = 0

    for tensor in request.inputs:
        shape = list(tensor.shape)
        try:
            shm_params = _extract_shared_memory_params(tensor.parameters)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        raw_content: Optional[bytes] = None
        if shm_params:
            region_name, region_size, region_offset = shm_params
            try:
                raw_content = shared_memory_registry.read(region_name, region_size, region_offset)
            except KeyError:
                raise ValueError(f"Unknown shared memory region {region_name}") from None
        elif raw_index < len(raw_inputs):
            raw_content = raw_inputs[raw_index]
            raw_index += 1

        if raw_content is not None:
            values = _deserialize_input_tensor(raw_content, shape, tensor.name)
        elif tensor.contents.bytes_contents:
            values = np.array(tensor.contents.bytes_contents, dtype=np.object_)
            if shape:
                try:
                    values = values.reshape(shape)
                except ValueError as exc:
                    raise ValueError(f"Invalid shape for input {tensor.name}: {exc}") from exc
        else:
            raise ValueError(f"No data provided for input {tensor.name}.")
        inputs[tensor.name] = values

    if "INPUT_IMAGE_URLS" not in inputs:
        raise ValueError("INPUT_IMAGE_URLS input is required.")

    image_urls = [_decode_string(value) for value in inputs["INPUT_IMAGE_URLS"].reshape(-1)]

    raw_merge_levels = inputs.get("MERGE_LEVELS")
    if raw_merge_levels is None:
        merge_levels: List[MergeLevel] = [settings.merge_level for _ in image_urls]
    else:
        merge_levels = []
        for value in raw_merge_levels.reshape(-1):
            merge_levels.append(_decode_string(value) or settings.merge_level)

    if len(image_urls) != len(merge_levels):
        raise ValueError("MERGE_LEVELS batch size does not match INPUT_IMAGE_URLS.")

    images: List[np.ndarray] = []
    for encoded in image_urls:
        try:
            images.append(_load_base64_image(encoded))
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    return images, merge_levels


def _build_infer_response(
    results: List[Tuple[List[List[List[float]]], List[str], List[float]]]
) -> service_pb2.ModelInferResponse:
    batch_size = len(results)
    output = np.empty((batch_size, 3), dtype=np.object_)
    for idx, (boxes, texts, confidences) in enumerate(results):
        output[idx, 0] = json.dumps(boxes).encode("utf-8")
        output[idx, 1] = json.dumps(texts).encode("utf-8")
        output[idx, 2] = json.dumps(confidences).encode("utf-8")

    serialized = serialize_byte_tensor(output)
    output_tensor = service_pb2.ModelInferResponse.InferOutputTensor(
        name="OUTPUT",
        datatype="BYTES",
        shape=[batch_size, 3],
    )
    return service_pb2.ModelInferResponse(
        model_name=settings.model_name,
        model_version=settings.model_version,
        outputs=[output_tensor],
        raw_output_contents=[serialized.tobytes()],
    )


def _infer_batch(
    images: List[np.ndarray], merge_levels: List[MergeLevel]
) -> List[Tuple[List[List[List[float]]], List[str], List[float]]]:
    model = _get_model()
    results: List[Tuple[List[List[List[float]]], List[str], List[float]]] = []
    for image, merge_level in zip(images, merge_levels):
        preds = model(image, merge_level=merge_level, visualize=False)
        height, width = image.shape[:2]
        results.append(_collect_predictions(preds, width, height))
    return results


class TritonOCRService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self) -> None:
        self.last_inference_ns: Optional[int] = None

    async def _run_model_infer(
        self,
        request: service_pb2.ModelInferRequest,
        context: grpc.ServicerContext,
        span_name: str,
    ) -> service_pb2.ModelInferResponse:
        REQUEST_COUNTER.labels("grpc").inc()
        start_time = time.perf_counter()
        results: List[Tuple[List[List[List[float]]], List[str], List[float]]] = []

        if not _is_model_loaded():
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")

        with tracer.start_as_current_span(span_name) as span:
            try:
                images, merge_levels = _extract_grpc_inputs(request)
            except ValueError as exc:
                span.record_exception(exc)
                span.set_status(Status(status_code=StatusCode.ERROR))
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

            span.set_attribute("ocr.batch_size", len(images))
            if len(images) > settings.max_batch_size:
                context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"Batch size {len(images)} exceeds limit {settings.max_batch_size}.",
                )

            try:
                results = await asyncio.to_thread(_infer_batch, images, merge_levels)
            except Exception as exc:  # pragma: no cover - surfaced to caller
                logger.exception("Model inference failed (gRPC).")
                span.record_exception(exc)
                span.set_status(Status(status_code=StatusCode.ERROR))
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

            response = _build_infer_response(results)
            serialized = response.raw_output_contents[0] if response.raw_output_contents else b""

            output_region: Optional[str] = None
            output_offset = 0
            output_byte_size: Optional[int] = None
            if request.outputs:
                output_lookup: Dict[str, service_pb2.ModelInferRequest.InferRequestedOutputTensor] = {
                    out.name: out for out in request.outputs
                }
                requested_output = output_lookup.get("OUTPUT")
                if requested_output is not None:
                    try:
                        shm_params = _extract_shared_memory_params(requested_output.parameters)
                    except ValueError as exc:
                        span.record_exception(exc)
                        span.set_status(Status(status_code=StatusCode.ERROR))
                        context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                    if shm_params:
                        output_region, output_byte_size, output_offset = shm_params

            if output_region is not None:
                if output_byte_size is None:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required for outputs")
                if len(serialized) > output_byte_size:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Shared memory region '{output_region}' is too small for output",
                    )
                try:
                    shared_memory_registry.write(output_region, serialized, output_offset)
                except KeyError:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown shared memory region {output_region}")
                except ValueError as exc:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                response.raw_output_contents.clear()

            self.last_inference_ns = time.time_ns()

        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels("grpc").observe(duration)
        BATCH_SIZE.labels("grpc").observe(len(results))
        _record_inference_success(int(duration * 1e9))
        return response

    async def ModelInfer(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        return await self._run_model_infer(request, context, "grpc_infer")

    async def ModelReady(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        ready = _is_model_loaded() and request.name in {settings.model_name, ""}
        return service_pb2.ModelReadyResponse(ready=ready)

    async def ServerReady(self, request, context):  # noqa: N802
        return service_pb2.ServerReadyResponse(ready=_is_model_loaded())

    async def ServerLive(self, request, context):  # noqa: N802
        return service_pb2.ServerLiveResponse(live=True)

    async def ServerMetadata(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        return service_pb2.ServerMetadataResponse(
            name=settings.model_name,
            version=settings.model_version,
            extensions=["model_repository", "metrics"],
        )

    async def ModelMetadata(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if request.name and request.name != settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        return _model_metadata_response()

    async def ModelConfig(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if request.name and request.name != settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        return model_config_pb2.ModelConfigResponse(config=_model_config_proto())

    async def ModelStatistics(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        with _model_state_lock:
            stats = service_pb2.ModelStatistics(
                name=settings.model_name,
                version=settings.model_version,
                last_inference=_model_state.last_inference_ns,
                inference_count=_model_state.inference_count,
                execution_count=_model_state.inference_count,
            )
            stats.inference_stats.success.count = _model_state.inference_count
            stats.inference_stats.success.ns = _model_state.success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    async def RepositoryIndex(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        entry = service_pb2.RepositoryIndexResponse.ModelIndex(
            name=settings.model_name,
            version=settings.model_version,
            state="READY" if _is_model_loaded() else "UNAVAILABLE",
        )
        return service_pb2.RepositoryIndexResponse(models=[entry])

    async def RepositoryModelLoad(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            _load_model()
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load model: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to load model")
        return service_pb2.RepositoryModelLoadResponse()

    async def RepositoryModelUnload(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        _unload_model()
        return service_pb2.RepositoryModelUnloadResponse()

    async def ModelStreamInfer(self, request_iterator, context):  # noqa: N802
        _authorize_grpc(context)
        try:
            async for request in request_iterator:
                response = await self._run_model_infer(request, context, "grpc_stream_infer")
                yield response
        except grpc.RpcError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Streaming inference failed.")
            context.abort(grpc.StatusCode.INTERNAL, "Streaming inference failed")

    async def TraceSetting(self, request, context):  # noqa: N802
        return service_pb2.TraceSettingResponse()

    async def LogSettings(self, request, context):  # noqa: N802
        return service_pb2.LogSettingsResponse()

    async def CudaSharedMemoryRegister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.register_cuda(
                request.name,
                bytes(request.raw_handle),
                int(request.device_id),
                int(request.byte_size),
                int(request.offset),
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:  # pragma: no cover - surfaced to caller
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return service_pb2.CudaSharedMemoryRegisterResponse()

    async def CudaSharedMemoryStatus(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        response = service_pb2.CudaSharedMemoryStatusResponse()
        regions = shared_memory_registry.cuda_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' not found")
        for region in regions:
            status = service_pb2.CudaSharedMemoryStatusResponse.RegionStatus(
                name=region.name,
                device_id=region.device_id,
                byte_size=region.byte_size,
            )
            response.regions[region.name].CopyFrom(status)
        return response

    async def CudaSharedMemoryUnregister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_cuda(request.name)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return service_pb2.CudaSharedMemoryUnregisterResponse()

    async def SystemSharedMemoryRegister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name or not request.key:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name and key are required")
        try:
            shared_memory_registry.register_system(
                request.name,
                request.key,
                int(request.byte_size),
                int(request.offset),
            )
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:  # pragma: no cover - surfaced to caller
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        return service_pb2.SystemSharedMemoryRegisterResponse()

    async def SystemSharedMemoryStatus(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        response = service_pb2.SystemSharedMemoryStatusResponse()
        regions = shared_memory_registry.system_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"System shared memory region '{request.name}' not found")
        for region in regions:
            status = service_pb2.SystemSharedMemoryStatusResponse.RegionStatus(
                name=region.name,
                key=region.key,
                offset=region.offset,
                byte_size=region.byte_size,
            )
            response.regions[region.name].CopyFrom(status)
        return response

    async def SystemSharedMemoryUnregister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_system(request.name)
        except ValueError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        return service_pb2.SystemSharedMemoryUnregisterResponse()


@app.on_event("startup")
async def _startup() -> None:
    _load_model()
    _start_metrics_server()
    await _start_grpc_server()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if grpc_server is not None:
        await grpc_server.stop(grace=2)


@app.get("/v1/health/live")
async def live() -> JSONResponse:
    return JSONResponse({"status": "live"})


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    is_ready = _current_model_ready()
    status_value = "ready" if is_ready else "unavailable"
    return JSONResponse({"status": status_value}, status_code=200 if is_ready else 503)


@app.get("/v1/metadata", dependencies=[Depends(require_auth)])
async def metadata() -> JSONResponse:
    model_info = {
        "id": "nvidia/nemotron-ocr-v1",
        "name": "nemotron-ocr-v1",
        "version": settings.model_version,
        "shortName": "nemoretriever-ocr-v1",
        "maxBatchSize": settings.max_batch_size,
    }
    return JSONResponse({"modelInfo": [model_info]})


@app.get("/metrics", dependencies=[Depends(require_auth)])
@app.get("/v2/metrics", dependencies=[Depends(require_auth)])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/infer", dependencies=[Depends(require_auth)])
async def infer(request: InferRequest) -> dict:
    if not request.input:
        raise HTTPException(status_code=400, detail="input must contain at least one image_url item")

    if len(request.input) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request.input)} exceeds limit {settings.max_batch_size}.",
        )

    if not _current_model_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    merge_levels = request.merge_levels or [settings.merge_level] * len(request.input)
    if len(merge_levels) != len(request.input):
        raise HTTPException(status_code=400, detail="merge_levels length must match input length.")

    images: List[np.ndarray] = []
    try:
        for item in request.input:
            images.append(_load_image(item.url))
    except (OSError, ValueError, ValidationError) as exc:
        logger.exception("Invalid image payload.")
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.RequestException as exc:
        logger.exception("Failed to fetch remote image.")
        raise HTTPException(status_code=502, detail=str(exc))

    REQUEST_COUNTER.labels("http").inc()
    start_time = time.perf_counter()
    with tracer.start_as_current_span("http_infer") as span:
        span.set_attribute("ocr.batch_size", len(images))
        success = False
        try:
            model = _get_model()
            results: List[InferResponseItem] = []
            for image, merge_level in zip(images, merge_levels):
                preds = await asyncio.to_thread(model, image, merge_level=merge_level, visualize=False)
                height, width = image.shape[:2]
                results.append(_format_predictions(preds, width, height))
            success = True
        except Exception as exc:  # pragma: no cover - surfaced to client
            span.record_exception(exc)
            span.set_status(Status(status_code=StatusCode.ERROR))
            logger.exception("Model inference failed.")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            duration = time.perf_counter() - start_time
            REQUEST_LATENCY.labels("http").observe(duration)
            BATCH_SIZE.labels("http").observe(len(images))
            if success:
                _record_inference_success(int(duration * 1e9))

    return {"data": results}


@app.get("/v2/health/live")
async def triton_live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready")
async def triton_ready() -> Dict[str, bool]:
    return {"ready": _current_model_ready()}


@app.get("/v2/models", dependencies=[Depends(require_auth)])
async def list_models() -> Dict[str, object]:
    state = "READY" if _current_model_ready() else "UNAVAILABLE"
    return {
        "models": [
            {
                "name": settings.model_name,
                "version": settings.model_version,
                "state": state,
            }
        ]
    }


@app.get("/v2/models/{model_name}", dependencies=[Depends(require_auth)])
async def model_metadata_http(model_name: str) -> Dict[str, object]:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    proto = _model_metadata_response()
    return json_format.MessageToDict(proto, preserving_proto_field_name=True)


@app.get("/v2/models/{model_name}/config", dependencies=[Depends(require_auth)])
async def model_config_http(model_name: str) -> Dict[str, object]:
    if model_name != settings.model_name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    proto = _model_config_proto()
    return json_format.MessageToDict(proto, preserving_proto_field_name=True)


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> Dict[str, bool]:
    is_ready = model_name == settings.model_name and _current_model_ready()
    return {"ready": is_ready}


async def _start_grpc_server() -> None:
    global grpc_server
    global _triton_service
    if grpc_server is not None:
        return

    server = grpc.aio.server()
    _triton_service = TritonOCRService()
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_triton_service, server)
    server.add_insecure_port(f"0.0.0.0:{settings.grpc_port}")
    await server.start()
    grpc_server = server
    logger.info("Started Triton-compatible gRPC server on :%d", settings.grpc_port)
    asyncio.create_task(server.wait_for_termination())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
