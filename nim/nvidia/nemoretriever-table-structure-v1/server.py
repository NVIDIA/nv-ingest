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
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from time import perf_counter
from typing import BinaryIO, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import grpc
from google.protobuf.json_format import MessageToDict
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from nemotron_table_structure_v1.model import define_model
from nemotron_table_structure_v1.utils import postprocess_preds_table_structure
from PIL import Image
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor, serialize_byte_tensor, triton_to_np_dtype

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None

logger = logging.getLogger("nemoretriever-table-structure-v1")
logging.basicConfig(level=logging.INFO)

OUTPUT_LABELS: List[str] = ["border", "cell", "row", "column", "header"]


class ServiceSettings(BaseSettings):
    model_name: str = Field("nemoretriever-table-structure-v1", alias="MODEL_ID")
    model_version: str = Field("1.6.0", alias="YOLOX_TABLE_STRUCTURE_TAG")
    max_batch_size_env: Optional[int] = Field(None, alias="NIM_TRITON_MAX_BATCH_SIZE")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int = Field(8002, alias="NIM_METRICS_API_PORT")
    triton_model_name: str = Field("yolox_ensemble", alias="TRITON_MODEL_NAME")
    triton_model_platform: str = Field("python", alias="TRITON_MODEL_PLATFORM")
    triton_config_path: Optional[str] = Field(None, alias="TRITON_MODEL_CONFIG_PATH")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_service_name: str = Field("nemoretriever-table-structure-v1", alias="NIM_OTEL_SERVICE_NAME")
    otel_traces_exporter: str = Field("otlp", alias="NIM_OTEL_TRACES_EXPORTER")
    otel_metrics_exporter: str = Field("otlp", alias="NIM_OTEL_METRICS_EXPORTER")
    otel_endpoint: str = Field("http://otel-collector:4318", alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    ngc_api_key: Optional[str] = Field(None, alias="NGC_API_KEY")
    nvidia_api_key: Optional[str] = Field(None, alias="NVIDIA_API_KEY")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    triton_log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    enable_model_control: bool = Field(True, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")

    class Config:
        extra = "ignore"


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
    def ensure_non_empty(cls, value: List[ImageInput]) -> List[ImageInput]:
        if len(value) == 0:
            raise ValueError("input must include at least one image")
        return value


@dataclass
class TensorSpec:
    name: str
    dtype: str
    shape: List[int]


@dataclass
class TritonModelConfig:
    name: str
    version: str
    platform: str
    max_batch_size: int
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


@dataclass
class ModelState:
    loaded: bool = True
    inference_count: int = 0
    last_inference_ns: int = 0
    success_ns_total: int = 0


@dataclass
class SystemSharedMemoryRegion:
    name: str
    key: str
    offset: int
    byte_size: int
    shm: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: BinaryIO | None = None

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
    device_id: int
    byte_size: int
    raw_handle: bytes
    offset: int
    pointer: Optional[int] = None


CUDA_IPC_FLAG = 1
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
_cudart: Optional[ctypes.CDLL] = None


def _get_cudart() -> ctypes.CDLL:
    global _cudart
    if _cudart is not None:
        return _cudart
    library_path = ctypes.util.find_library("cudart")
    candidates = [library_path] if library_path else []
    candidates.extend(["libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "libcudart.dylib"])
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


def _check_cuda_status(status: int, op: str) -> None:
    if status != 0:
        raise RuntimeError(f"{op} failed with status {status}")


def _open_cuda_memory(raw_handle: bytes, device_id: int) -> Tuple[ctypes.CDLL, int]:
    if not raw_handle:
        raise ValueError("raw_handle is required for CUDA shared memory registration")
    cudart = _get_cudart()
    _check_cuda_status(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")
    handle_buffer = ctypes.create_string_buffer(raw_handle)
    device_pointer = ctypes.c_void_p()
    _check_cuda_status(
        cudart.cudaIpcOpenMemHandle(
            ctypes.byref(device_pointer),
            ctypes.cast(handle_buffer, ctypes.c_void_p),
            CUDA_IPC_FLAG,
        ),
        "cudaIpcOpenMemHandle",
    )
    if device_pointer.value is None:
        raise RuntimeError("Failed to map CUDA shared memory handle")
    return cudart, int(device_pointer.value)


def _close_cuda_memory(pointer: Optional[int]) -> None:
    if pointer is None:
        return
    try:
        cudart = _get_cudart()
        _check_cuda_status(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)), "cudaIpcCloseMemHandle")
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to close CUDA shared memory handle", exc_info=True)


def _read_cuda_shared_memory(region: CudaSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if offset < 0:
        raise ValueError("shared_memory_offset must be non-negative")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + byte_size
    if start < region.offset or end > region.offset + region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    cudart = _get_cudart()
    _check_cuda_status(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
    host_buffer = ctypes.create_string_buffer(byte_size)
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda_status(
        cudart.cudaMemcpy(
            ctypes.cast(host_buffer, ctypes.c_void_p),
            device_ptr,
            ctypes.c_size_t(byte_size),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "cudaMemcpy (device to host)",
    )
    return bytes(host_buffer.raw)


def _write_cuda_shared_memory(region: CudaSharedMemoryRegion, data: bytes, offset: int, requested_byte_size: int) -> None:
    if requested_byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if offset < 0:
        raise ValueError("shared_memory_offset must be non-negative")
    if len(data) > requested_byte_size:
        raise ValueError("Shared memory target too small for data")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + requested_byte_size
    if start < region.offset or end > region.offset + region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    cudart = _get_cudart()
    _check_cuda_status(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
    host_buffer = ctypes.create_string_buffer(data, len(data))
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda_status(
        cudart.cudaMemcpy(
            device_ptr,
            ctypes.cast(host_buffer, ctypes.c_void_p),
            ctypes.c_size_t(len(data)),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "cudaMemcpy (host to device)",
    )


class AuthValidator:
    def __init__(self, tokens: Iterable[str], require: bool) -> None:
        self._tokens = {token for token in tokens if token}
        self._require = require

    def _extract_token(self, header_value: Optional[str]) -> Optional[str]:
        if header_value is None:
            return None
        stripped = header_value.strip()
        if stripped == "":
            return None
        parts = stripped.split(" ", maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            return token if token != "" else None
        return stripped

    def validate(self, header_value: Optional[str], api_key_header: Optional[str] = None) -> bool:
        token = self._extract_token(header_value)
        if token is None:
            token = self._extract_token(api_key_header)
        if not self._require:
            return True
        if len(self._tokens) > 0:
            return token in self._tokens
        return token is not None


class MetricsRecorder:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.registry = registry
        self.requests = Counter(
            "nim_requests_total",
            "Total inference requests",
            ["protocol", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "nim_request_latency_seconds",
            "Request latency in seconds",
            ["protocol"],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self.batch_size = Histogram(
            "nim_request_batch_size",
            "Batch sizes observed",
            ["protocol"],
            registry=self.registry,
            buckets=tuple(float(size) for size in [1, 2, 4, 8, 16, 32, 64]),
        )

    def observe(self, protocol: str, status: str, duration: float, batch_size: int) -> None:
        self.requests.labels(protocol=protocol, status=status).inc()
        self.latency.labels(protocol=protocol).observe(duration)
        self.batch_size.labels(protocol=protocol).observe(float(batch_size))


def _read_max_batch_size(settings: ServiceSettings) -> int:
    if settings.triton_config_path is not None and os.path.exists(settings.triton_config_path):
        try:
            with open(settings.triton_config_path, "r", encoding="utf-8") as file:
                for line in file:
                    stripped = line.strip()
                    if stripped.startswith("max_batch_size"):
                        _, value = stripped.split(":", maxsplit=1)
                        parsed = int(value.strip())
                        if parsed > 0:
                            return parsed
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to parse Triton config at %s: %s", settings.triton_config_path, exc)
    if settings.max_batch_size_env is not None and settings.max_batch_size_env > 0:
        return settings.max_batch_size_env
    return 32


def _setup_tracer(settings: ServiceSettings) -> Optional[trace.Tracer]:
    if not settings.enable_otel:
        return None
    if trace is None or OTLPSpanExporter is None or TracerProvider is None or BatchSpanProcessor is None:
        logger.warning("OpenTelemetry requested but exporters are unavailable; skipping OTEL setup")
        return None
    if settings.otel_traces_exporter.lower() != "otlp":
        return None
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.model_name)


def _decode_image(data_url: str) -> np.ndarray:
    prefix, encoded = data_url.split("base64,", maxsplit=1)
    if not prefix.startswith("data:image"):
        raise ValueError("only data:image/* base64 URLs are supported")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("invalid base64 payload") from exc
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError("invalid image payload") from exc


def _predict(model: torch.nn.Module, images: List[np.ndarray], score_threshold: Optional[float]) -> List[Dict[str, Dict[str, List[Dict[str, float]]]]]:
    processed = [model.preprocess(image) for image in images]
    batch = torch.stack(processed)
    original_sizes = np.array([image.shape[:2] for image in images])
    with torch.inference_mode():
        preds = model(batch, original_sizes)
    threshold = score_threshold if score_threshold is not None else getattr(model, "threshold", 0.1)
    results: List[Dict[str, Dict[str, List[Dict[str, float]]]]] = []
    for pred in preds:
        boxes, labels, scores = postprocess_preds_table_structure(
            pred,
            threshold=threshold,
            class_labels=getattr(model, "labels", OUTPUT_LABELS),
            reorder=True,
        )
        bounding_boxes: Dict[str, List[Dict[str, float]]] = {label: [] for label in OUTPUT_LABELS}
        for box, label_idx, score in zip(boxes, labels, scores):
            label_name = model.labels[int(label_idx)]
            if label_name not in bounding_boxes:
                continue
            x_min, y_min, x_max, y_max = (float(coord) for coord in box.tolist())
            bounding_boxes[label_name].append(
                {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "confidence": float(score),
                }
            )
        results.append({"bounding_boxes": bounding_boxes})
    return results


def _json_for_grpc(prediction: Dict[str, Dict[str, List[Dict[str, float]]]]) -> str:
    serialized = {
        label: [
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"], box["confidence"]]
            for box in boxes
        ]
        for label, boxes in prediction["bounding_boxes"].items()
    }
    return json.dumps(serialized, separators=(",", ":"))


def _model_metadata_response(config: TritonModelConfig) -> service_pb2.ModelMetadataResponse:
    response = service_pb2.ModelMetadataResponse(
        name=config.name,
        platform=config.platform,
        versions=[config.version],
    )
    for spec in config.inputs:
        tensor = response.inputs.add()
        tensor.name = spec.name
        tensor.datatype = spec.dtype
        tensor.shape.extend(spec.shape)
    for spec in config.outputs:
        tensor = response.outputs.add()
        tensor.name = spec.name
        tensor.datatype = spec.dtype
        tensor.shape.extend(spec.shape)
    return response


def _model_config_response(config: TritonModelConfig) -> service_pb2.ModelConfigResponse:
    response = service_pb2.ModelConfigResponse()
    response.config.name = config.name
    response.config.platform = config.platform
    response.config.max_batch_size = config.max_batch_size
    dtype_map = {"BYTES": model_config_pb2.TYPE_STRING, "FP32": model_config_pb2.TYPE_FP32}
    for spec in config.inputs:
        inp = response.config.input.add()
        inp.name = spec.name
        inp.data_type = dtype_map.get(spec.dtype, model_config_pb2.TYPE_STRING)
        inp.dims.extend(spec.shape)
    for spec in config.outputs:
        out = response.config.output.add()
        out.name = spec.name
        out.data_type = dtype_map.get(spec.dtype, model_config_pb2.TYPE_STRING)
        out.dims.extend(spec.shape)
    return response


def _parse_shared_memory_params(parameters: Mapping[str, service_pb2.InferParameter]) -> Tuple[Optional[str], Optional[int], int]:
    region = parameters["shared_memory_region"].string_param if "shared_memory_region" in parameters else None
    byte_size = parameters["shared_memory_byte_size"].int64_param if "shared_memory_byte_size" in parameters else None
    offset = parameters["shared_memory_offset"].int64_param if "shared_memory_offset" in parameters else 0
    return region or None, byte_size, offset


async def _register_system_shared_memory(
    name: str,
    key: str,
    offset: int,
    byte_size: int,
    lock: asyncio.Lock,
    registry: Dict[str, SystemSharedMemoryRegion],
    cuda_registry: Dict[str, CudaSharedMemoryRegion],
) -> None:
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    async with lock:
        if name in registry or name in cuda_registry:
            raise FileExistsError(f"Region {name} already registered")
    shm_obj: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: BinaryIO | None = None
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
            name=name,
            key=key,
            offset=offset,
            byte_size=byte_size,
            shm=shm_obj,
            mmap_obj=mmap_obj,
            file_handle=file_handle,
        )
        async with lock:
            registry[name] = region
    except Exception:
        if shm_obj is not None:
            shm_obj.close()
        if mmap_obj is not None:
            mmap_obj.close()
        if file_handle is not None:
            file_handle.close()
        raise


async def _unregister_system_shared_memory(
    name: str,
    lock: asyncio.Lock,
    registry: Dict[str, SystemSharedMemoryRegion],
) -> None:
    async with lock:
        region = registry.pop(name, None)
    if region is not None:
        region.close()


async def _register_cuda_shared_memory(
    name: str,
    raw_handle: bytes,
    device_id: int,
    byte_size: int,
    lock: asyncio.Lock,
    registry: Dict[str, CudaSharedMemoryRegion],
    system_registry: Dict[str, SystemSharedMemoryRegion],
    offset: int,
) -> None:
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    pointer: Optional[int] = None
    try:
        _, pointer = _open_cuda_memory(raw_handle, device_id)
        region = CudaSharedMemoryRegion(
            name=name,
            device_id=device_id,
            byte_size=byte_size,
            raw_handle=raw_handle,
            offset=offset,
            pointer=pointer,
        )
        async with lock:
            if name in registry or name in system_registry:
                raise FileExistsError(f"Region {name} already registered")
            registry[name] = region
    except Exception:
        _close_cuda_memory(pointer)
        raise


async def _unregister_cuda_shared_memory(
    name: str,
    lock: asyncio.Lock,
    registry: Dict[str, CudaSharedMemoryRegion],
) -> None:
    async with lock:
        region = registry.pop(name, None)
    if region is not None:
        _close_cuda_memory(region.pointer)


async def _read_shared_memory(
    region_name: str,
    byte_size: int,
    offset: int,
    context: grpc.aio.ServicerContext,
    lock: asyncio.Lock,
    system_registry: Dict[str, SystemSharedMemoryRegion],
    cuda_registry: Dict[str, CudaSharedMemoryRegion],
) -> bytes:
    async with lock:
        system_region = system_registry.get(region_name)
        cuda_region = cuda_registry.get(region_name)
    if system_region is None and cuda_region is None:
        raise KeyError(region_name)
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if offset < 0:
        raise ValueError("shared_memory_offset must be non-negative")
    if cuda_region is not None:
        return _read_cuda_shared_memory(cuda_region, byte_size, offset)
    start = system_region.offset + offset
    end = start + byte_size
    if start < 0 or end > system_region.offset + system_region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    return bytes(system_region.buffer[start:end])


async def _write_shared_memory(
    region_name: str,
    data: bytes,
    offset: int,
    byte_size: int,
    lock: asyncio.Lock,
    system_registry: Dict[str, SystemSharedMemoryRegion],
    cuda_registry: Dict[str, CudaSharedMemoryRegion],
) -> None:
    async with lock:
        system_region = system_registry.get(region_name)
        cuda_region = cuda_registry.get(region_name)
    if system_region is None and cuda_region is None:
        raise KeyError(region_name)
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if offset < 0:
        raise ValueError("shared_memory_offset must be non-negative")
    if len(data) > byte_size:
        raise ValueError("Shared memory target too small for data")
    if cuda_region is not None:
        _write_cuda_shared_memory(cuda_region, data, offset, byte_size)
        return
    start = system_region.offset + offset
    end = start + byte_size
    if start < 0 or end > system_region.offset + system_region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    system_region.buffer[start:start + len(data)] = data


class ModelRunner:
    def __init__(
        self,
        model_getter: Callable[[], Optional[torch.nn.Module]],
        triton_config: TritonModelConfig,
        metrics: MetricsRecorder,
        tracer: Optional[trace.Tracer],
        semaphore: Optional[asyncio.Semaphore],
        model_state: ModelState,
        model_lock: asyncio.Lock,
    ) -> None:
        self._model_getter = model_getter
        self._config = triton_config
        self._metrics = metrics
        self._tracer = tracer
        self._semaphore = semaphore
        self._model_state = model_state
        self._model_lock = model_lock

    @asynccontextmanager
    async def _limit(self) -> Iterable[None]:
        if self._semaphore is None:
            yield
            return
        async with self._semaphore:
            yield

    async def infer(self, images: List[np.ndarray], threshold: Optional[float]) -> List[Dict[str, Dict[str, List[Dict[str, float]]]]]:
        async with self._limit():
            loop = asyncio.get_running_loop()
            async with self._model_lock:
                is_loaded = self._model_state.loaded
                model = self._model_getter()
            if not is_loaded or model is None:
                raise RuntimeError("Model is not loaded")
            if self._tracer is None:
                return await loop.run_in_executor(None, _predict, model, images, threshold)
            with self._tracer.start_as_current_span("table_structure.infer") as span:
                span.set_attribute("batch_size", len(images))
                span.set_attribute("triton.max_batch_size", self._config.max_batch_size)
                return await loop.run_in_executor(None, _predict, model, images, threshold)


class TritonInferenceService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(
        self,
        runner: ModelRunner,
        triton_config: TritonModelConfig,
        auth_validator: AuthValidator,
        model_state: ModelState,
        metrics: MetricsRecorder,
        model_lock: asyncio.Lock,
        model_state_lock: asyncio.Lock,
        shared_memory_lock: asyncio.Lock,
        system_shared_memory: Dict[str, SystemSharedMemoryRegion],
        cuda_shared_memory: Dict[str, CudaSharedMemoryRegion],
    ) -> None:
        self._runner = runner
        self._config = triton_config
        self._auth_validator = auth_validator
        self._model_state = model_state
        self._metrics = metrics
        self._model_lock = model_lock
        self._model_state_lock = model_state_lock
        self._shared_memory_lock = shared_memory_lock
        self._system_shared_memory = system_shared_memory
        self._cuda_shared_memory = cuda_shared_memory

    async def _ensure_auth(self, context: grpc.aio.ServicerContext, method_name: str) -> None:
        metadata = {item.key.lower(): item.value for item in context.invocation_metadata()}
        header_value = metadata.get("authorization") or metadata.get("grpcgateway-authorization")
        if self._auth_validator.validate(header_value):
            return
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, f"Unauthorized access to {method_name}")

    async def ServerLive(self, request: service_pb2.ServerLiveRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerLiveResponse:  # noqa: N802
        return service_pb2.ServerLiveResponse(live=True)

    async def ServerReady(self, request: service_pb2.ServerReadyRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerReadyResponse:  # noqa: N802
        return service_pb2.ServerReadyResponse(ready=self._model_state.loaded)

    async def ModelReady(self, request: service_pb2.ModelReadyRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelReadyResponse:  # noqa: N802
        model_name = request.name or self._config.name
        ready = self._model_state.loaded and model_name == self._config.name
        return service_pb2.ModelReadyResponse(ready=ready)

    async def ServerMetadata(self, request: service_pb2.ServerMetadataRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerMetadataResponse:  # noqa: N802
        return service_pb2.ServerMetadataResponse(
            name=self._config.name,
            version=self._config.version,
            extensions=[
                "model_repository",
                "model_repository_unload",
                "model_repository_load",
                "model_statistics",
                "system_shared_memory",
                "cuda_shared_memory",
            ],
        )

    async def ModelMetadata(self, request: service_pb2.ModelMetadataRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelMetadataResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelMetadata")
        model_name = request.name or self._config.name
        if model_name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Model {model_name} not found")
        return _model_metadata_response(self._config)

    async def RepositoryIndex(self, request: service_pb2.RepositoryIndexRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryIndexResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryIndex")
        response = service_pb2.RepositoryIndexResponse()
        model_index = response.models.add()
        model_index.name = self._config.name
        model_index.version = self._config.version
        model_index.state = "READY" if self._model_state.loaded else "UNAVAILABLE"
        model_index.reason = "" if self._model_state.loaded else "Model is unloaded"
        return response

    async def RepositoryModelLoad(self, request: service_pb2.RepositoryModelLoadRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryModelLoadResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryModelLoad")
        if not settings.enable_model_control:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            async with self._model_lock:
                if app.state.model is None:
                    loop = asyncio.get_running_loop()
                    app.state.model = await loop.run_in_executor(None, _load_model)
                self._model_state.loaded = True
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load model: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to load model")
        return service_pb2.RepositoryModelLoadResponse()

    async def RepositoryModelUnload(self, request: service_pb2.RepositoryModelUnloadRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryModelUnloadResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryModelUnload")
        if not settings.enable_model_control:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        async with self._model_lock:
            self._model_state.loaded = False
            app.state.model = None
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to clear CUDA cache on unload", exc_info=True)
        return service_pb2.RepositoryModelUnloadResponse()

    async def _decode_inputs(self, request: service_pb2.ModelInferRequest, context: grpc.aio.ServicerContext) -> Dict[str, np.ndarray]:
        decoded: Dict[str, np.ndarray] = {}
        raw_contents = list(request.raw_input_contents)
        raw_index = 0
        for tensor in request.inputs:
            shape = tuple(int(dim) for dim in tensor.shape)
            raw_content = raw_contents[raw_index] if raw_index < len(raw_contents) else None
            region_name, region_size, region_offset = _parse_shared_memory_params(tensor.parameters)
            if raw_content is not None:
                raw_index += 1
            elif region_name is not None:
                if region_size is None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required")
                try:
                    raw_content = await _read_shared_memory(
                        region_name,
                        int(region_size),
                        int(region_offset),
                        context,
                        self._shared_memory_lock,
                        self._system_shared_memory,
                        self._cuda_shared_memory,
                    )
                except KeyError:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {region_name}")
                except ValueError as exc:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except RuntimeError as exc:
                    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            if tensor.datatype == "BYTES":
                if raw_content is not None:
                    array = deserialize_bytes_tensor(raw_content)
                else:
                    array = np.array(list(tensor.contents.bytes_contents), dtype=np.object_)
                if len(shape) > 0 and np.prod(shape) == array.size:
                    array = array.reshape(shape)
            else:
                dtype = triton_to_np_dtype(tensor.datatype)
                if raw_content is not None:
                    array = np.frombuffer(raw_content, dtype=dtype)
                else:
                    field = tensor.contents
                    if dtype == np.float32:
                        array = np.array(list(field.fp32_contents), dtype=dtype)
                    elif dtype == np.float64:
                        array = np.array(list(field.fp64_contents), dtype=dtype)
                    elif dtype == np.int64:
                        array = np.array(list(field.int64_contents), dtype=dtype)
                    elif dtype == np.int32:
                        array = np.array(list(field.int_contents), dtype=dtype)
                    else:
                        array = np.array(list(field.uint_contents), dtype=dtype)
                if len(shape) > 0:
                    array = array.reshape(shape)
            decoded[tensor.name] = array
        return decoded

    def _extract_threshold(self, inputs: Dict[str, np.ndarray]) -> Optional[float]:
        if "THRESHOLDS" not in inputs:
            return None
        thresholds = inputs["THRESHOLDS"]
        if thresholds.size == 0:
            return None
        flat = thresholds.reshape(-1)
        return float(flat[0])

    async def _run_infer_request(
        self,
        request: service_pb2.ModelInferRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.ModelInferResponse:
        if not self._model_state.loaded:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        decoded_inputs = await self._decode_inputs(request, context)
        if "INPUT_IMAGES" not in decoded_inputs:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "INPUT_IMAGES tensor is required")
        images_field = decoded_inputs["INPUT_IMAGES"]
        images_flat = images_field.reshape(-1)
        batch_size = images_flat.shape[0]
        if batch_size > self._config.max_batch_size:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Batch size {batch_size} exceeds limit {self._config.max_batch_size}",
            )
        try:
            images = [_decode_image(image.decode("utf-8") if isinstance(image, bytes) else str(image)) for image in images_flat]
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        threshold_override = self._extract_threshold(decoded_inputs)
        started = perf_counter()
        try:
            predictions = await self._runner.infer(images, threshold_override)
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise
        except Exception as exc:  # pylint: disable=broad-except
            await context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {exc}")
            raise
        payload_bytes = np.array([_json_for_grpc(pred).encode("utf-8") for pred in predictions], dtype=np.object_)
        response = service_pb2.ModelInferResponse()
        response.model_name = request.model_name or self._config.name
        response.model_version = self._config.version
        if request.id:
            response.id = request.id
        output = response.outputs.add()
        output.name = self._config.outputs[0].name
        output.datatype = self._config.outputs[0].dtype
        output.shape.extend([payload_bytes.shape[0]])
        output_region: Optional[str] = None
        output_offset = 0
        output_byte_size: Optional[int] = None
        if request.outputs:
            region_name, region_size, region_offset = _parse_shared_memory_params(request.outputs[0].parameters)
            if region_name:
                if region_size is None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required for outputs")
                output_region = region_name
                output_offset = int(region_offset)
                output_byte_size = int(region_size)
        serialized_output = serialize_byte_tensor(payload_bytes)
        if output_region is not None and output_byte_size is not None:
            try:
                await _write_shared_memory(
                    output_region,
                    serialized_output,
                    output_offset,
                    output_byte_size,
                    self._shared_memory_lock,
                    self._system_shared_memory,
                    self._cuda_shared_memory,
                )
            except KeyError:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown shared memory region {output_region}")
            except ValueError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            except Exception as exc:  # pragma: no cover - surfaced to caller
                logger.exception("Failed to write shared memory output: %s", exc)
                await context.abort(grpc.StatusCode.INTERNAL, "Failed to write shared memory output")
        else:
            response.raw_output_contents.append(serialized_output)
        duration = perf_counter() - started
        self._metrics.observe("grpc", "success", duration, batch_size)
        await _record_inference_success(int(duration * 1e9))
        return response

    async def ModelConfig(self, request: service_pb2.ModelConfigRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelConfigResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelConfig")
        return _model_config_response(self._config)

    async def ModelInfer(self, request: service_pb2.ModelInferRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelInferResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelInfer")
        return await self._run_infer_request(request, context)

    async def ModelStreamInfer(self, request_iterator, context):  # noqa: N802
        await self._ensure_auth(context, "ModelStreamInfer")
        try:
            async for request in request_iterator:
                response = await self._run_infer_request(request, context)
                yield response
        except grpc.RpcError:
            raise
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Streaming inference failed: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Streaming inference failed")

    async def ModelStatistics(self, request: service_pb2.ModelStatisticsRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelStatisticsResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelStatistics")
        requested_name = request.name or self._config.name
        if requested_name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Model {requested_name} not found")
        async with self._model_state_lock:
            stats = service_pb2.ModelStatistics(
                name=self._config.name,
                version=self._config.version,
                last_inference=self._model_state.last_inference_ns,
                inference_count=self._model_state.inference_count,
                execution_count=self._model_state.inference_count,
            )
            stats.inference_stats.success.count = self._model_state.inference_count
            stats.inference_stats.success.ns = self._model_state.success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    async def SystemSharedMemoryRegister(self, request: service_pb2.SystemSharedMemoryRegisterRequest, context: grpc.aio.ServicerContext) -> service_pb2.SystemSharedMemoryRegisterResponse:  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryRegister")
        if not request.name or not request.key:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name and key are required")
        try:
            await _register_system_shared_memory(
                request.name,
                request.key,
                int(request.offset),
                int(request.byte_size),
                self._shared_memory_lock,
                self._system_shared_memory,
                self._cuda_shared_memory,
            )
        except FileExistsError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("System shared memory registration failed: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to register shared memory")
        return service_pb2.SystemSharedMemoryRegisterResponse()

    async def SystemSharedMemoryStatus(self, request: service_pb2.SystemSharedMemoryStatusRequest, context: grpc.aio.ServicerContext) -> service_pb2.SystemSharedMemoryStatusResponse:  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryStatus")
        response = service_pb2.SystemSharedMemoryStatusResponse()
        async with self._shared_memory_lock:
            if request.name:
                if request.name not in self._system_shared_memory:
                    await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown region {request.name}")
                region_names = [request.name]
            else:
                region_names = list(self._system_shared_memory.keys())
            for region_name in region_names:
                region = self._system_shared_memory[region_name]
                status = service_pb2.SystemSharedMemoryStatusResponse.RegionStatus(
                    name=region.name,
                    key=region.key,
                    offset=region.offset,
                    byte_size=region.byte_size,
                )
                response.regions[region.name].CopyFrom(status)
        return response

    async def SystemSharedMemoryUnregister(self, request: service_pb2.SystemSharedMemoryUnregisterRequest, context: grpc.aio.ServicerContext) -> service_pb2.SystemSharedMemoryUnregisterResponse:  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryUnregister")
        if not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        async with self._shared_memory_lock:
            if request.name not in self._system_shared_memory:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown region {request.name}")
        await _unregister_system_shared_memory(request.name, self._shared_memory_lock, self._system_shared_memory)
        return service_pb2.SystemSharedMemoryUnregisterResponse()

    async def CudaSharedMemoryRegister(self, request: service_pb2.CudaSharedMemoryRegisterRequest, context: grpc.aio.ServicerContext) -> service_pb2.CudaSharedMemoryRegisterResponse:  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryRegister")
        if not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            await _register_cuda_shared_memory(
                request.name,
                bytes(request.raw_handle),
                int(request.device_id),
                int(request.byte_size),
                self._shared_memory_lock,
                self._cuda_shared_memory,
                self._system_shared_memory,
                int(request.offset),
            )
        except FileExistsError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        return service_pb2.CudaSharedMemoryRegisterResponse()

    async def CudaSharedMemoryStatus(self, request: service_pb2.CudaSharedMemoryStatusRequest, context: grpc.aio.ServicerContext) -> service_pb2.CudaSharedMemoryStatusResponse:  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryStatus")
        response = service_pb2.CudaSharedMemoryStatusResponse()
        async with self._shared_memory_lock:
            if request.name:
                if request.name not in self._cuda_shared_memory:
                    await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown CUDA region {request.name}")
                regions = [request.name]
            else:
                regions = list(self._cuda_shared_memory.keys())
            for region_name in regions:
                region = self._cuda_shared_memory[region_name]
                status = service_pb2.CudaSharedMemoryStatusResponse.RegionStatus(
                    name=region.name,
                    device_id=region.device_id,
                    byte_size=region.byte_size,
                )
                response.regions[region.name].CopyFrom(status)
        return response

    async def CudaSharedMemoryUnregister(self, request: service_pb2.CudaSharedMemoryUnregisterRequest, context: grpc.aio.ServicerContext) -> service_pb2.CudaSharedMemoryUnregisterResponse:  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryUnregister")
        if not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        async with self._shared_memory_lock:
            if request.name not in self._cuda_shared_memory:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown CUDA region {request.name}")
        await _unregister_cuda_shared_memory(request.name, self._shared_memory_lock, self._cuda_shared_memory)
        return service_pb2.CudaSharedMemoryUnregisterResponse()


settings = ServiceSettings()
logging.getLogger().setLevel(settings.log_level.upper())
if settings.triton_log_verbose > 0 and logging.getLogger().level > logging.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)

max_batch_size = _read_max_batch_size(settings)
triton_config = TritonModelConfig(
    name=settings.triton_model_name,
    version=settings.model_version,
    platform=settings.triton_model_platform,
    max_batch_size=max_batch_size,
    inputs=[
        TensorSpec(name="INPUT_IMAGES", dtype="BYTES", shape=[-1]),
        TensorSpec(name="THRESHOLDS", dtype="FP32", shape=[-1, 2]),
    ],
    outputs=[TensorSpec(name="OUTPUT", dtype="BYTES", shape=[-1])],
)

tracer = _setup_tracer(settings)
auth_validator = AuthValidator(
    tokens=[settings.ngc_api_key or "", settings.nvidia_api_key or ""],
    require=settings.require_auth,
)

registry = CollectorRegistry()
metrics = MetricsRecorder(registry)
inference_semaphore = asyncio.Semaphore(settings.rate_limit) if settings.rate_limit is not None and settings.rate_limit > 0 else None
model_state = ModelState()
model_state_lock = asyncio.Lock()
model_lock = asyncio.Lock()
shared_memory_lock = asyncio.Lock()
system_shared_memory: Dict[str, SystemSharedMemoryRegion] = {}
cuda_shared_memory: Dict[str, CudaSharedMemoryRegion] = {}


async def _record_inference_success(duration_ns: int) -> None:
    async with model_state_lock:
        model_state.inference_count += 1
        model_state.success_ns_total += duration_ns
        model_state.last_inference_ns = time.time_ns()


def _load_model() -> torch.nn.Module:
    model = define_model("table_structure_v1", verbose=False)
    model.eval()
    logger.info(
        "Loaded %s (version=%s) with threshold %.3f and batch limit %d",
        settings.model_name,
        settings.model_version,
        getattr(model, "threshold", 0.0),
        triton_config.max_batch_size,
    )
    return model


app = FastAPI(title=settings.model_name, version=settings.model_version)
app.state.model = _load_model()
runner = ModelRunner(
    lambda: app.state.model,
    triton_config,
    metrics,
    tracer,
    inference_semaphore,
    model_state,
    model_lock,
)
grpc_service = TritonInferenceService(
    runner,
    triton_config,
    auth_validator,
    model_state,
    metrics,
    model_lock,
    model_state_lock,
    shared_memory_lock,
    system_shared_memory,
    cuda_shared_memory,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterable[None]:
    start_http_server(settings.metrics_port, registry=metrics.registry)
    server = grpc.aio.server(interceptors=[])
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(grpc_service, server)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    await server.start()
    app.state.grpc_server = server
    try:
        yield
    finally:
        model_state.loaded = False
        app.state.model = None
        await server.stop(grace=None)


app.router.lifespan_context = lifespan


def _is_public_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    if normalized in {"/", "/v1/health/live", "/v1/health/ready", "/metrics"}:
        return True
    if normalized.startswith("/v2/health/"):
        return True
    if normalized.startswith("/v2/models/") and normalized.endswith("/ready"):
        return True
    return False


@app.middleware("http")
async def _log_and_auth(request: Request, call_next) -> Response:
    logger.info("%s %s", request.method, request.url.path)
    if not _is_public_path(request.url.path):
        if not auth_validator.validate(request.headers.get("authorization"), request.headers.get("x-api-key")):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    response = await call_next(request)
    return response


@app.get("/v1/health/live")
def live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v1/health/ready")
def ready() -> Dict[str, bool]:
    return {"ready": model_state.loaded}


@app.get("/v1/models")
def list_models() -> Dict[str, object]:
    return {"object": "list", "data": [{"id": settings.model_name}]}


@app.get("/v1/metadata")
def metadata() -> Dict[str, object]:
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
                "maxBatchSize": triton_config.max_batch_size,
            }
        ],
    }


@app.get("/metrics")
def metrics_endpoint() -> PlainTextResponse:
    content = generate_latest(metrics.registry)
    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")


@app.get("/v2/health/live")
def triton_live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready")
def triton_ready() -> Dict[str, bool]:
    return {"ready": model_state.loaded}


@app.get("/v2/models/{model_name}")
def model_metadata_http(model_name: str) -> JSONResponse:
    if model_name != triton_config.name:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    response = _model_metadata_response(triton_config)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/config")
def model_config_http(model_name: str) -> JSONResponse:
    if model_name != triton_config.name:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    response = _model_config_response(triton_config)
    return JSONResponse(MessageToDict(response, preserving_proto_field_name=True))


@app.get("/v2/models/{model_name}/ready")
def model_ready_http(model_name: str) -> Dict[str, bool]:
    is_ready = model_name == triton_config.name and model_state.loaded
    return {"ready": is_ready}


@app.post("/v1/infer")
async def infer(request_body: InferRequest) -> Dict[str, List[Dict[str, Dict[str, List[Dict[str, float]]]]]]:
    batch_size = len(request_body.input)
    if batch_size > triton_config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {batch_size} exceeds limit {triton_config.max_batch_size}",
        )
    if not model_state.loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        images = [_decode_image(item.url) for item in request_body.input]
    except ValueError as exc:
        logger.error("Failed to decode input image: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    started = perf_counter()
    try:
        predictions = await runner.infer(images, None)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    duration = perf_counter() - started
    metrics.observe("http", "success", duration, batch_size)
    await _record_inference_success(int(duration * 1e9))
    return {"data": predictions}


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "nemoretriever-table-structure-v1 HTTP"}, status_code=200)
