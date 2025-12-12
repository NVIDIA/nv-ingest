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
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import grpc
import numpy as np
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf import json_format
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import serialize_byte_tensor, triton_to_np_dtype

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    Status = None
    StatusCode = None

try:
    from paddleocr import PaddleOCRVL
except ImportError as exc:  # pragma: no cover - surfaced at startup
    msg = (
        "paddleocr is missing. Install PaddlePaddle GPU + paddleocr[doc-parser] "
        "(see README.md) before running the server."
    )
    raise RuntimeError(msg) from exc


class Settings(BaseSettings):
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_TRITON_GRPC_PORT")
    metrics_port: int = Field(8002, alias="NIM_TRITON_METRICS_PORT")
    max_batch_size: int = Field(2, alias="NIM_TRITON_MAX_BATCH_SIZE")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    triton_log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    request_timeout_seconds: float = Field(30.0, alias="REQUEST_TIMEOUT_SECONDS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    use_layout_detection: bool = Field(True, alias="USE_LAYOUT_DETECTION")
    use_chart_recognition: bool = Field(True, alias="USE_CHART_RECOGNITION")
    format_block_content: bool = Field(False, alias="FORMAT_BLOCK_CONTENT")
    merge_layout_blocks: bool = Field(True, alias="MERGE_LAYOUT_BLOCKS")
    vl_rec_backend: Optional[str] = Field(None, alias="VL_REC_BACKEND")
    vl_rec_server_url: Optional[str] = Field(None, alias="VL_REC_SERVER_URL")
    model_id: str = Field("baidu/paddleocr", alias="MODEL_ID")
    model_version: str = Field("1.5.0", alias="MODEL_VERSION")
    short_name: str = Field("paddleocr", alias="MODEL_SHORT_NAME")
    model_name: str = Field("paddle", alias="OCR_MODEL_NAME")
    auth_token: Optional[str] = Field(None, alias="NGC_API_KEY")
    alt_auth_token: Optional[str] = Field(None, alias="NVIDIA_API_KEY")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_service_name: str = Field("paddleocr", alias="NIM_OTEL_SERVICE_NAME")
    otel_endpoint: Optional[str] = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class InferInput(BaseModel):
    type: Literal["image_url"]
    url: str | HttpUrl


class InferRequest(BaseModel):
    input: List[InferInput]
    merge_levels: List[str] | None = None


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


@dataclass
class ModelState:
    loaded: bool = False
    inference_count: int = 0
    last_inference_ns: int = 0
    success_ns_total: int = 0


@dataclass
class SystemSharedMemoryRegion:
    name: str
    key: str
    offset: int
    byte_size: int
    shm: Optional[shared_memory.SharedMemory] = None
    mmap_obj: Optional[mmap.mmap] = None
    file_handle: Optional[Any] = None

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
                logger.debug("Failed to close shared memory file handle", exc_info=True)


@dataclass
class CudaSharedMemoryRegion:
    name: str
    raw_handle: bytes
    device_id: int
    byte_size: int
    offset: int
    pointer: Optional[int] = None


class SharedMemoryRegistry:
    CUDA_IPC_FLAG = 1
    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self._system: Dict[str, SystemSharedMemoryRegion] = {}
        self._cuda: Dict[str, CudaSharedMemoryRegion] = {}
        self._lock = threading.Lock()
        self._cudart: Optional[ctypes.CDLL] = None

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
    def _check_cuda(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with status {status}")

    def _resolve_region(self, name: str) -> Tuple[str, SystemSharedMemoryRegion | CudaSharedMemoryRegion]:
        with self._lock:
            if name in self._system:
                return "system", self._system[name]
            if name in self._cuda:
                return "cuda", self._cuda[name]
        raise KeyError(f"Shared memory region '{name}' is not registered")

    @staticmethod
    def _validate_bounds(start: int, length: int, limit: int) -> None:
        if start < 0:
            raise ValueError("shared_memory_offset must be non-negative")
        if length <= 0:
            raise ValueError("shared_memory_byte_size must be positive")
        if start + length > limit:
            raise ValueError("Requested shared memory range exceeds registered byte_size")

    def register_system(self, name: str, key: str, offset: int, byte_size: int) -> None:
        if not name:
            raise ValueError("name is required")
        if not key:
            raise ValueError("key is required")
        if byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._lock:
            if name in self._system or name in self._cuda:
                raise FileExistsError(f"Shared memory region '{name}' is already registered")
        shm_obj: Optional[shared_memory.SharedMemory] = None
        mmap_obj: Optional[mmap.mmap] = None
        file_handle: Optional[Any] = None
        buffer_size: int
        try:
            try:
                shm_obj = shared_memory.SharedMemory(name=key, create=False)
                buffer_size = shm_obj.size
            except FileNotFoundError:
                path = key[7:] if key.startswith("file://") else key
                file_handle = open(path, "r+b")
                buffer_size = os.fstat(file_handle.fileno()).st_size
                mmap_obj = mmap.mmap(file_handle.fileno(), length=buffer_size)
            if offset + byte_size > buffer_size:
                raise ValueError("Requested shared memory range exceeds buffer size")
            region = SystemSharedMemoryRegion(name, key, offset, byte_size, shm=shm_obj, mmap_obj=mmap_obj, file_handle=file_handle)
            with self._lock:
                self._system[name] = region
        except Exception:
            if shm_obj is not None:
                shm_obj.close()
            if mmap_obj is not None:
                mmap_obj.close()
            if file_handle is not None:
                try:
                    file_handle.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.debug("Failed to close file handle", exc_info=True)
            raise

    def system_regions(self) -> List[SystemSharedMemoryRegion]:
        with self._lock:
            return list(self._system.values())

    def unregister_system(self, name: str) -> None:
        with self._lock:
            region = self._system.pop(name, None)
        if region is None:
            raise FileNotFoundError(f"Shared memory region '{name}' is not registered")
        region.close()

    def register_cuda(self, name: str, raw_handle: bytes, device_id: int, byte_size: int, offset: int) -> None:
        if not name:
            raise ValueError("name is required")
        if not raw_handle:
            raise ValueError("raw_handle is required")
        if byte_size <= 0:
            raise ValueError("byte_size must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._lock:
            if name in self._system or name in self._cuda:
                raise FileExistsError(f"Shared memory region '{name}' is already registered")

        cudart = self._get_cudart()
        pointer = ctypes.c_void_p()
        self._check_cuda(cudart.cudaSetDevice(device_id), "cudaSetDevice")
        handle_buffer = ctypes.create_string_buffer(raw_handle)
        status = cudart.cudaIpcOpenMemHandle(
            ctypes.byref(pointer),
            ctypes.cast(handle_buffer, ctypes.c_void_p),
            self.CUDA_IPC_FLAG,
        )
        self._check_cuda(status, "cudaIpcOpenMemHandle")
        region = CudaSharedMemoryRegion(
            name=name,
            raw_handle=bytes(raw_handle),
            device_id=device_id,
            byte_size=byte_size,
            offset=offset,
            pointer=int(pointer.value),
        )
        with self._lock:
            self._cuda[name] = region

    def cuda_regions(self) -> List[CudaSharedMemoryRegion]:
        with self._lock:
            return list(self._cuda.values())

    def unregister_cuda(self, name: str) -> None:
        with self._lock:
            region = self._cuda.pop(name, None)
        if region is None:
            raise FileNotFoundError(f"Shared memory region '{name}' is not registered")
        if region.pointer is None:
            return
        try:
            cudart = self._get_cudart()
            self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
            self._check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(region.pointer)), "cudaIpcCloseMemHandle")
        finally:
            region.pointer = None

    def read(self, name: str, byte_size: int, offset: int) -> bytes:
        kind, region = self._resolve_region(name)
        start = region.offset + offset
        self._validate_bounds(start, byte_size, region.byte_size + region.offset)
        if kind == "system":
            buffer = region.buffer
            return bytes(buffer[start : start + byte_size])
        cudart = self._get_cudart()
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        dest = ctypes.create_string_buffer(byte_size)
        status = cudart.cudaMemcpy(
            ctypes.cast(dest, ctypes.c_void_p),
            ctypes.c_void_p(region.pointer + start),
            ctypes.c_size_t(byte_size),
            self.CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        self._check_cuda(status, "cudaMemcpy")
        return bytes(dest.raw)

    def write(self, name: str, payload: bytes, offset: int) -> None:
        if not payload:
            return
        kind, region = self._resolve_region(name)
        start = region.offset + offset
        self._validate_bounds(start, len(payload), region.byte_size + region.offset)
        if kind == "system":
            buffer = region.buffer
            buffer[start : start + len(payload)] = payload
            return
        cudart = self._get_cudart()
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        dest_ptr = ctypes.c_void_p(region.pointer + start)
        src = (ctypes.c_char * len(payload)).from_buffer_copy(payload)
        status = cudart.cudaMemcpy(
            dest_ptr,
            ctypes.cast(src, ctypes.c_void_p),
            ctypes.c_size_t(len(payload)),
            self.CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        self._check_cuda(status, "cudaMemcpy")

settings = Settings()
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("paddleocr-nim")

shared_memory_registry = SharedMemoryRegistry()

def _create_paddleocr_instance() -> PaddleOCRVL:
    return PaddleOCRVL(
        use_layout_detection=settings.use_layout_detection,
        use_chart_recognition=settings.use_chart_recognition,
        format_block_content=settings.format_block_content,
        vl_rec_backend=settings.vl_rec_backend,
        vl_rec_server_url=settings.vl_rec_server_url,
        merge_layout_blocks=settings.merge_layout_blocks,
    )


_paddleocr_instance: Optional[PaddleOCRVL] = _create_paddleocr_instance()
_model_state = ModelState(loaded=True)
_state_lock = asyncio.Lock()

app = FastAPI(title=settings.model_id, version=settings.model_version)
grpc_server: Optional[grpc.aio.Server] = None
_triton_service: Optional["TritonPaddleService"] = None

REQUEST_COUNTER = Counter("nim_paddleocr_requests_total", "Total inference requests", ["protocol"])
REQUEST_LATENCY = Histogram("nim_paddleocr_request_latency_seconds", "Latency for inference requests", ["protocol"])
BATCH_SIZE = Histogram("nim_paddleocr_batch_size", "Observed batch sizes", ["protocol"])
_METRICS_STARTED = False

_rate_limiter = asyncio.Semaphore(settings.rate_limit) if settings.rate_limit else None


def _configure_tracer() -> Optional[trace.Tracer]:
    endpoint = settings.otel_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not settings.enable_otel and not endpoint:
        return None
    if trace is None or OTLPSpanExporter is None or TracerProvider is None or BatchSpanProcessor is None:
        logger.warning("OpenTelemetry requested but not available; skipping OTEL setup.")
        return None
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


tracer = _configure_tracer()


def _expected_token() -> Optional[str]:
    token = settings.auth_token or settings.alt_auth_token
    return token if token else None


def _validate_bearer(value: Optional[str], on_error) -> None:
    if not settings.require_auth:
        return
    token = _expected_token()
    if token is None:
        return
    if not value:
        on_error("Missing bearer token.")
        return
    scheme, _, credential = value.partition(" ")
    if scheme.lower() != "bearer" or credential.strip() == "":
        on_error("Invalid authorization scheme.")
        return
    if credential.strip() != token:
        on_error("Invalid bearer token.")


async def _record_inference_success(duration_seconds: float) -> None:
    async with _state_lock:
        _model_state.inference_count += 1
        _model_state.last_inference_ns = time.time_ns()
        _model_state.success_ns_total += int(duration_seconds * 1e9)


async def _load_model(force: bool = False) -> None:
    global _paddleocr_instance
    async with _state_lock:
        if _model_state.loaded and _paddleocr_instance is not None and not force:
            return
        _model_state.loaded = False
        try:
            instance = await asyncio.to_thread(_create_paddleocr_instance)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load PaddleOCR-VL model: %s", exc)
            raise RuntimeError("Failed to load model") from exc
        _paddleocr_instance = instance
        _model_state.loaded = True


async def _unload_model() -> None:
    global _paddleocr_instance
    async with _state_lock:
        _paddleocr_instance = None
        _model_state.loaded = False


def _current_model_ready() -> bool:
    return _model_state.loaded and _paddleocr_instance is not None


async def require_auth(
    authorization: Optional[str] = Header(default=None),
    api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    def _raise(detail: str) -> None:
        raise HTTPException(status_code=401, detail=detail)

    header_value = authorization or api_key
    _validate_bearer(header_value, _raise)


def _authorize_grpc(context: grpc.ServicerContext) -> None:
    def _abort(detail: str) -> None:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, detail)

    metadata = {md.key.lower(): md.value for md in context.invocation_metadata()}
    auth_value = metadata.get("authorization")
    _validate_bearer(auth_value, _abort)


def _start_metrics_server() -> None:
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    if settings.require_auth:
        logger.info("Metrics auth enabled; skipping standalone metrics server.")
        _METRICS_STARTED = True
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


def _to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1]


def _load_image(url: str, timeout: float) -> np.ndarray:
    if url.startswith("data:"):
        return _to_bgr(Image.open(io.BytesIO(_decode_data_url(url))))

    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return _to_bgr(Image.open(io.BytesIO(response.content)))

    return _to_bgr(Image.open(url))


def _normalize_points(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> List[Point]:
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain four values.")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions for normalization.")

    x_min, y_min, x_max, y_max = [float(val) for val in bbox]
    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))

    def _clip(value: float) -> float:
        return max(0.0, min(1.0, value))

    return [
        Point(x=_clip(x_min / width), y=_clip(y_min / height)),
        Point(x=_clip(x_max / width), y=_clip(y_min / height)),
        Point(x=_clip(x_max / width), y=_clip(y_max / height)),
        Point(x=_clip(x_min / width), y=_clip(y_max / height)),
    ]


def _format_result(result: Any) -> InferResponseItem:
    raw_width = result.get("width")
    raw_height = result.get("height")
    width = int(raw_width) if raw_width is not None else 0
    height = int(raw_height) if raw_height is not None else 0
    if (width == 0 or height == 0) and "doc_preprocessor_res" in result:
        try:
            output_img = result["doc_preprocessor_res"]["output_img"]
            height = int(output_img.shape[0])
            width = int(output_img.shape[1])
        except Exception:
            logger.debug("Falling back to doc_preprocessor_res failed for image dims.")
    detections: List[TextDetection] = []

    parsing_blocks = result.get("parsing_res_list") or []

    for block in parsing_blocks:
        try:
            text = str(getattr(block, "content", "") or "").strip()
            bbox = getattr(block, "bbox", None)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Skipping malformed block: %s", exc)
            continue

        if text == "" or bbox is None:
            continue

        try:
            points = _normalize_points(bbox, width, height)
        except ValueError as exc:
            logger.debug("Skipping block with invalid bbox: %s", exc)
            continue

        detections.append(
            TextDetection(
                bounding_box=BoundingBox(points=points),
                text_prediction=TextPrediction(text=text, confidence=1.0),
            )
        )

    return InferResponseItem(text_detections=detections)


def _get_paddleocr() -> PaddleOCRVL:
    if _paddleocr_instance is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return _paddleocr_instance


def _infer(images: List[np.ndarray]) -> List[InferResponseItem]:
    try:
        results = _get_paddleocr().predict(
            images,
            use_layout_detection=settings.use_layout_detection,
            use_chart_recognition=settings.use_chart_recognition,
            format_block_content=settings.format_block_content,
            merge_layout_blocks=settings.merge_layout_blocks,
        )
    except Exception as exc:
        logger.exception("PaddleOCR-VL inference failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [_format_result(item) for item in results]


def _denormalize_chw(image: np.ndarray) -> np.ndarray:
    # Input is CHW float32 normalized as (x/255 - mean)/std (RGB order).
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor with shape (3, H, W).")
    chw = image.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (chw * std + mean) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))[:, :, ::-1]  # convert to BGR


def _collect_triplets(item: InferResponseItem) -> Tuple[List[List[List[float]]], List[str], List[float]]:
    boxes: List[List[List[float]]] = []
    texts: List[str] = []
    confidences: List[float] = []
    for detection in item.text_detections:
        points = detection.bounding_box.points
        boxes.append([[float(pt.x), float(pt.y)] for pt in points])
        texts.append(detection.text_prediction.text)
        confidences.append(float(detection.text_prediction.confidence))
    return boxes, texts, confidences


def _build_infer_response(
    items: List[InferResponseItem], request_id: str = ""
) -> service_pb2.ModelInferResponse:
    batch_size = len(items)
    payload = np.empty((3, batch_size), dtype=np.object_)
    for idx, item in enumerate(items):
        boxes, texts, confidences = _collect_triplets(item)
        payload[0, idx] = json.dumps(boxes).encode("utf-8")
        payload[1, idx] = json.dumps(texts).encode("utf-8")
        payload[2, idx] = json.dumps(confidences).encode("utf-8")

    serialized = serialize_byte_tensor(payload)
    output_tensor = service_pb2.ModelInferResponse.InferOutputTensor(
        name="output",
        datatype="BYTES",
        shape=[3, batch_size],
    )
    return service_pb2.ModelInferResponse(
        model_name=settings.model_name,
        model_version=settings.model_version,
        outputs=[output_tensor],
        raw_output_contents=[serialized.tobytes()],
        id=request_id,
    )


def _model_config_proto() -> model_config_pb2.ModelConfig:
    return model_config_pb2.ModelConfig(
        name=settings.model_name,
        platform="python",
        max_batch_size=settings.max_batch_size,
        input=[
            model_config_pb2.ModelInput(name="input", data_type=model_config_pb2.TYPE_FP32, dims=[-1, 3, -1, -1]),
        ],
        output=[
            model_config_pb2.ModelOutput(name="output", data_type=model_config_pb2.TYPE_BYTES, dims=[3, -1]),
        ],
    )


def _model_metadata_response() -> service_pb2.ModelMetadataResponse:
    return service_pb2.ModelMetadataResponse(
        name=settings.model_name,
        versions=[settings.model_version],
        platform="python",
        inputs=[
            service_pb2.ModelMetadataResponse.TensorMetadata(
                name="input",
                datatype="FP32",
                shape=[-1, 3, -1, -1],
            ),
        ],
        outputs=[
            service_pb2.ModelMetadataResponse.TensorMetadata(
                name="output",
                datatype="BYTES",
                shape=[3, -1],
            )
        ],
    )


def _model_ready() -> bool:
    return _current_model_ready()


@asynccontextmanager
async def _limit_concurrency():
    if _rate_limiter is None:
        yield
        return
    async with _rate_limiter:
        yield


def _parse_shared_memory_params(
    parameters: Mapping[str, service_pb2.ModelInferRequest.InferParameter],
) -> Tuple[Optional[str], Optional[int], int]:
    region_param = parameters.get("shared_memory_region")
    region_name = None
    if region_param is not None and region_param.WhichOneof("parameter_choice") == "string_param":
        region_name = region_param.string_param or None

    byte_size_param = parameters.get("shared_memory_byte_size")
    byte_size = None
    if byte_size_param is not None and byte_size_param.WhichOneof("parameter_choice"):
        byte_size = int(byte_size_param.int64_param or byte_size_param.uint64_param)

    offset_param = parameters.get("shared_memory_offset")
    offset = 0
    if offset_param is not None and offset_param.WhichOneof("parameter_choice"):
        offset = int(offset_param.int64_param or offset_param.uint64_param)
        if offset < 0:
            raise ValueError("shared_memory_offset must be non-negative")
    return region_name, byte_size, offset


def _resolve_output_region(
    outputs: List[service_pb2.ModelInferRequest.InferRequestedOutputTensor],
) -> Tuple[Optional[str], Optional[int], int]:
    if not outputs:
        return None, None, 0
    output_tensor = outputs[0]
    return _parse_shared_memory_params(output_tensor.parameters)


def _write_output_to_shared_memory(name: str, payload: bytes, offset: int) -> None:
    try:
        shared_memory_registry.write(name, payload, offset)
        return
    except KeyError as exc:
        raise FileNotFoundError(str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise exc


def _deserialize_input_tensor(tensor: service_pb2.ModelInferRequest.InferInputTensor, raw: Optional[bytes]) -> np.ndarray:
    if raw is None and tensor.contents.raw_contents:
        raw = tensor.contents.raw_contents
    if raw is None and tensor.contents.fp32_contents:
        array = np.array(tensor.contents.fp32_contents, dtype=np.float32)
        try:
            return array.reshape(tuple(tensor.shape))
        except ValueError as exc:
            raise ValueError(f"Invalid shape for input {tensor.name}: {exc}") from exc
    if raw is None:
        raise ValueError(f"No data provided for input {tensor.name}.")

    dtype = triton_to_np_dtype(tensor.datatype)
    array = np.frombuffer(raw, dtype=dtype)
    try:
        return array.reshape(tuple(tensor.shape))
    except ValueError as exc:
        raise ValueError(f"Invalid shape for input {tensor.name}: {exc}") from exc


def _extract_grpc_images(request: service_pb2.ModelInferRequest) -> List[np.ndarray]:
    if not request.inputs:
        raise ValueError("No inputs provided.")
    name_to_index: Dict[str, int] = {inp.name.lower(): idx for idx, inp in enumerate(request.inputs)}
    if "input" not in name_to_index:
        raise ValueError("Expected input tensor named 'input'.")

    idx = name_to_index["input"]
    tensor = request.inputs[idx]
    raw = request.raw_input_contents[idx] if idx < len(request.raw_input_contents) else None
    try:
        region_name, region_size, region_offset = _parse_shared_memory_params(tensor.parameters)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    payload = raw
    if payload is None and region_name is not None:
        if region_size is None or region_size <= 0:
            raise ValueError("shared_memory_byte_size must be positive when using shared memory")
        try:
            payload = shared_memory_registry.read(region_name, int(region_size), int(region_offset))
        except KeyError as exc:
            raise FileNotFoundError(str(exc)) from exc
    values = _deserialize_input_tensor(tensor, payload)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("input must have shape [batch, 3, H, W].")
    images: List[np.ndarray] = []
    for sample in values:
        images.append(_denormalize_chw(sample))
    return images


class TritonPaddleService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self) -> None:
        pass

    async def _run_infer_request(self, request, context, *, skip_auth: bool = False):  # noqa: N802
        if not skip_auth:
            _authorize_grpc(context)
        REQUEST_COUNTER.labels("grpc").inc()
        start_time = time.perf_counter()
        with tracer.start_as_current_span("grpc_infer") if tracer else nullcontext():
            async with _limit_concurrency():
                if not _current_model_ready():
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded.")
                if request.model_name and request.model_name not in {"", settings.model_name}:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")
                try:
                    images = _extract_grpc_images(request)
                except FileNotFoundError as exc:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
                except ValueError as exc:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except RuntimeError as exc:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

                batch_size = len(images)
                if batch_size == 0:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No images provided.")
                if batch_size > settings.max_batch_size:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"Batch size {batch_size} exceeds limit {settings.max_batch_size}.",
                    )

                try:
                    results = await asyncio.to_thread(_infer, images)
                except HTTPException as exc:
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.INTERNAL, exc.detail)
                except Exception as exc:  # pragma: no cover - surfaced to caller
                    logger.exception("Model inference failed (gRPC).")
                    REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                    context.abort(grpc.StatusCode.INTERNAL, str(exc))

        response = _build_infer_response(results, request.id)
        output_region, output_size, output_offset = _resolve_output_region(list(request.outputs))
        raw_payload = response.raw_output_contents[0] if response.raw_output_contents else b""
        if output_region is not None:
            if output_size is None or output_size <= 0:
                REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size must be positive for outputs")
            if raw_payload and len(raw_payload) > output_size:
                REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Shared memory region '{output_region}' is too small for output",
                )
            try:
                _write_output_to_shared_memory(output_region, raw_payload, int(output_offset))
                response.raw_output_contents[:] = []
            except FileNotFoundError as exc:
                REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            except ValueError as exc:
                REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:  # pragma: no cover - defensive surface
                REQUEST_LATENCY.labels("grpc").observe(time.perf_counter() - start_time)
                context.abort(grpc.StatusCode.INTERNAL, str(exc))

        duration = time.perf_counter() - start_time
        REQUEST_LATENCY.labels("grpc").observe(duration)
        BATCH_SIZE.labels("grpc").observe(float(len(results)))
        await _record_inference_success(duration)
        return response

    async def ModelInfer(self, request, context):  # noqa: N802
        return await self._run_infer_request(request, context)

    async def ModelReady(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        ready = _current_model_ready() and (request.name in {settings.model_name, ""})
        return service_pb2.ModelReadyResponse(ready=ready)

    async def ServerReady(self, request, context):  # noqa: N802
        return service_pb2.ServerReadyResponse(ready=_current_model_ready())

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
        if request.name and request.name != settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        async with _state_lock:
            inference_count = _model_state.inference_count
            last_inference = _model_state.last_inference_ns
            success_ns_total = _model_state.success_ns_total
        stats = service_pb2.ModelStatistics(
            name=settings.model_name,
            version=settings.model_version,
            last_inference=last_inference,
            inference_count=inference_count,
            execution_count=inference_count,
        )
        stats.inference_stats.success.count = inference_count
        stats.inference_stats.success.ns = success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    async def RepositoryIndex(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        entry = service_pb2.RepositoryIndexResponse.ModelIndex(
            name=settings.model_name,
            version=settings.model_version,
            state="READY" if _current_model_ready() else "UNAVAILABLE",
        )
        return service_pb2.RepositoryIndexResponse(models=[entry])

    async def RepositoryModelLoad(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model control disabled")
        if request.model_name and request.model_name != settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")
        try:
            await _load_model()
        except RuntimeError as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load model on demand: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to load model")
        return service_pb2.RepositoryModelLoadResponse()

    async def RepositoryModelUnload(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model control disabled")
        if request.model_name and request.model_name != settings.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")
        try:
            await _unload_model()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to unload model: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to unload model")
        return service_pb2.RepositoryModelUnloadResponse()

    async def ModelStreamInfer(self, request_iterator, context):  # noqa: N802
        _authorize_grpc(context)
        try:
            async for request in request_iterator:
                response = await self._run_infer_request(request, context, skip_auth=True)
                yield response
        except grpc.RpcError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Streaming inference failed: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Streaming inference failed")

    async def TraceSetting(self, request, context):  # noqa: N802
        return service_pb2.TraceSettingResponse()

    async def LogSettings(self, request, context):  # noqa: N802
        return service_pb2.LogSettingsResponse()

    async def CudaSharedMemoryRegister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        try:
            shared_memory_registry.register_cuda(
                request.name,
                bytes(request.raw_handle),
                int(request.device_id),
                int(request.byte_size),
                int(request.offset),
            )
            return service_pb2.CudaSharedMemoryRegisterResponse()
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to register CUDA shared memory: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to register CUDA shared memory")

    async def CudaSharedMemoryStatus(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        response = service_pb2.CudaSharedMemoryStatusResponse()
        regions = shared_memory_registry.cuda_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' is not registered")
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.device_id = region.device_id
            status.byte_size = region.byte_size
        return response

    async def CudaSharedMemoryUnregister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_cuda(request.name)
            return service_pb2.CudaSharedMemoryUnregisterResponse()
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to unregister CUDA shared memory: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to unregister CUDA shared memory")

    async def SystemSharedMemoryRegister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        try:
            shared_memory_registry.register_system(
                request.name,
                request.key,
                int(request.offset),
                int(request.byte_size),
            )
            return service_pb2.SystemSharedMemoryRegisterResponse()
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to register system shared memory: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to register system shared memory")

    async def SystemSharedMemoryStatus(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        response = service_pb2.SystemSharedMemoryStatusResponse()
        regions = shared_memory_registry.system_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                context.abort(grpc.StatusCode.NOT_FOUND, f"System shared memory region '{request.name}' is not registered")
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.key = region.key
            status.offset = region.offset
            status.byte_size = region.byte_size
        return response

    async def SystemSharedMemoryUnregister(self, request, context):  # noqa: N802
        _authorize_grpc(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_system(request.name)
            return service_pb2.SystemSharedMemoryUnregisterResponse()
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to unregister system shared memory: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to unregister system shared memory")


@app.middleware("http")
async def _http_tracing(request: Request, call_next):
    if tracer is None:
        return await call_next(request)

    span_name = f"http {request.method.lower()} {request.url.path}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - surfaced to caller
            span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                span.set_status(Status(status_code=StatusCode.ERROR))
            raise
        span.set_attribute("http.status_code", response.status_code)
        if response.status_code >= 400 and Status is not None and StatusCode is not None:
            span.set_status(Status(status_code=StatusCode.ERROR))
        return response


@app.on_event("startup")
async def _startup() -> None:
    _start_metrics_server()
    await _start_grpc_server()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if grpc_server is not None:
        await grpc_server.stop(grace=2)
    await _unload_model()


@app.get("/v1/health/live")
async def live() -> JSONResponse:
    return JSONResponse({"status": "live"}, status_code=200)


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    is_ready = _model_ready()
    return JSONResponse({"status": "ready" if is_ready else "unavailable"}, status_code=200 if is_ready else 503)


@app.get("/v1/metadata", dependencies=[Depends(require_auth)])
async def metadata() -> JSONResponse:
    model_info = {
        "id": settings.model_id,
        "name": settings.model_id,
        "version": settings.model_version,
        "shortName": settings.short_name,
        "maxBatchSize": settings.max_batch_size,
    }
    return JSONResponse({"modelInfo": [model_info]}, status_code=200)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": settings.model_id}, status_code=200)


@app.get("/metrics", dependencies=[Depends(require_auth)])
@app.get("/v2/metrics", dependencies=[Depends(require_auth)])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/infer", dependencies=[Depends(require_auth)])
async def infer(request: InferRequest) -> dict:
    if len(request.input) == 0:
        raise HTTPException(status_code=400, detail="input must contain at least one image_url item")

    if len(request.input) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(request.input)} exceeds limit {settings.max_batch_size}.",
        )

    if not _model_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        images = [_load_image(item.url, settings.request_timeout_seconds) for item in request.input]
    except (OSError, ValueError, ValidationError) as exc:
        logger.exception("Invalid image payload.")
        raise HTTPException(status_code=400, detail=str(exc))
    except requests.RequestException as exc:
        logger.exception("Failed to fetch remote image.")
        raise HTTPException(status_code=502, detail=str(exc))

    REQUEST_COUNTER.labels("http").inc()
    start_time = time.perf_counter()
    async with _limit_concurrency():
        with tracer.start_as_current_span("http_infer") if tracer else nullcontext():
            responses = await asyncio.to_thread(_infer, images)
    duration = time.perf_counter() - start_time
    REQUEST_LATENCY.labels("http").observe(duration)
    BATCH_SIZE.labels("http").observe(float(len(images)))
    await _record_inference_success(duration)
    return {"data": [resp.model_dump() for resp in responses]}


@app.get("/v2/health/live")
async def triton_live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready")
async def triton_ready() -> Dict[str, bool]:
    return {"ready": _model_ready()}


@app.get("/v2/models", dependencies=[Depends(require_auth)])
async def list_models() -> Dict[str, object]:
    state = "READY" if _model_ready() else "UNAVAILABLE"
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
    is_ready = model_name == settings.model_name and _model_ready()
    return {"ready": is_ready}


async def _start_grpc_server() -> None:
    global grpc_server
    global _triton_service
    if grpc_server is not None:
        return

    server = grpc.aio.server()
    _triton_service = TritonPaddleService()
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_triton_service, server)
    server.add_insecure_port(f"0.0.0.0:{settings.grpc_port}")
    await server.start()
    grpc_server = server
    logger.info("Started Triton-compatible gRPC server on :%d", settings.grpc_port)
    asyncio.create_task(server.wait_for_termination())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
