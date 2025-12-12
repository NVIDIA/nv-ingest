from __future__ import annotations

import asyncio
import logging
import os
import time
import ctypes
import ctypes.util
import mmap
import threading
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal, Mapping, Optional

import grpc
import numpy as np
from multiprocessing import shared_memory
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from google.protobuf import json_format
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor, triton_to_np_dtype
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None


logger = logging.getLogger("llama-3.2-nv-rerankqa-1b-v2")
logging.basicConfig(level=logging.INFO)

TruncationMode = Literal["NONE", "END", "START"]

QUERY_INPUT_NAME = "QUERY"
PASSAGES_INPUT_NAME = "PASSAGES"
TRUNCATE_INPUT_NAME = "TRUNCATE"
OUTPUT_NAME = "SCORES"


class ServiceSettings(BaseSettings):
    model_id: str = Field("nvidia/llama-3.2-nv-rerankqa-1b-v2", alias="MODEL_ID")
    model_version: str = Field("1.8.0", alias="MODEL_VERSION")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_TRITON_GRPC_PORT")
    metrics_port: int = Field(8002, alias="NIM_TRITON_METRICS_PORT")
    max_batch_size_env: Optional[int] = Field(64, alias="NIM_TRITON_MAX_BATCH_SIZE")
    max_sequence_length: Optional[int] = Field(None, alias="MAX_SEQUENCE_LENGTH")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    enable_model_control: bool = Field(False, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    auth_token: Optional[str] = Field(None, alias="NGC_API_KEY")
    alt_auth_token: Optional[str] = Field(None, alias="NVIDIA_API_KEY")
    fallback_auth_token: Optional[str] = Field(None, alias="NIM_NGC_API_KEY")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_service_name: str = Field("llama-3.2-nv-rerankqa-1b-v2", alias="NIM_OTEL_SERVICE_NAME")
    otel_traces_exporter: str = Field("otlp", alias="NIM_OTEL_TRACES_EXPORTER")
    otel_endpoint: str = Field("http://otel-collector:4318", alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    triton_config_path: Optional[str] = Field(None, alias="TRITON_MODEL_CONFIG_PATH")

    class Config:
        extra = "ignore"

    @property
    def resolved_auth_tokens(self) -> List[str]:
        tokens = [self.auth_token, self.alt_auth_token, self.fallback_auth_token]
        return [token for token in tokens if token]


class TextBlock(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("text must not be empty")
        return value


class Passage(TextBlock):
    metadata: Optional[Dict[str, Any]] = None


class RankingRequest(BaseModel):
    model: Optional[str] = None
    query: TextBlock
    passages: List[Passage]
    truncate: TruncationMode = "END"

    @field_validator("passages")
    @classmethod
    def validate_passages(cls, value: List[Passage]) -> List[Passage]:
        if not value:
            raise ValueError("passages must contain at least one entry")
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
                file_handle.seek(0, os.SEEK_END)
                buffer_size = file_handle.tell()
                file_handle.seek(0)
                mmap_obj = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_WRITE)
            if offset + byte_size > buffer_size:
                raise ValueError("Requested shared memory window exceeds the available buffer")
            region = SystemSharedMemoryRegion(
                name=name,
                key=key,
                offset=offset,
                byte_size=byte_size,
                shm=shm_obj,
                mmap_obj=mmap_obj,
                file_handle=file_handle,
            )
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
                    logger.debug("Failed to close shared memory file handle", exc_info=True)
            raise

    def unregister_system(self, name: str) -> None:
        with self._lock:
            region = self._system.pop(name, None)
        if region is None:
            raise FileNotFoundError(f"System shared memory region '{name}' is not registered")
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

        try:
            with self._lock:
                if name in self._system or name in self._cuda:
                    raise FileExistsError(f"Shared memory region '{name}' is already registered")
                self._cuda[name] = region
        except Exception:
            if device_pointer.value is not None:
                self._check_cuda(cudart.cudaIpcCloseMemHandle(device_pointer), "cudaIpcCloseMemHandle")
            raise

    def unregister_cuda(self, name: str) -> None:
        with self._lock:
            region = self._cuda.pop(name, None)
        if region is None:
            raise FileNotFoundError(f"CUDA shared memory region '{name}' is not registered")
        if region.pointer is None:
            return
        cudart = self._get_cudart()
        device_pointer = ctypes.c_void_p(region.pointer)
        self._check_cuda(cudart.cudaIpcCloseMemHandle(device_pointer), "cudaIpcCloseMemHandle")

    def system_regions(self) -> List[SystemSharedMemoryRegion]:
        with self._lock:
            return list(self._system.values())

    def cuda_regions(self) -> List[CudaSharedMemoryRegion]:
        with self._lock:
            return list(self._cuda.values())

    def _resolve_region(self, name: str) -> tuple[str, SystemSharedMemoryRegion | CudaSharedMemoryRegion]:
        with self._lock:
            if name in self._system:
                return "system", self._system[name]
            if name in self._cuda:
                return "cuda", self._cuda[name]
        raise KeyError(f"Shared memory region '{name}' is not registered")

    @staticmethod
    def _read_system(region: SystemSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        start = region.offset + offset
        end = start + byte_size
        if start < region.offset or end > region.offset + region.byte_size:
            raise ValueError("Requested read exceeds registered system shared memory bounds")
        return bytes(region.buffer[start:end])

    @staticmethod
    def _write_system(region: SystemSharedMemoryRegion, data: bytes, offset: int) -> None:
        start = region.offset + offset
        end = start + len(data)
        if start < region.offset or end > region.offset + region.byte_size:
            raise ValueError("Requested write exceeds registered system shared memory bounds")
        region.buffer[start:end] = data

    def _read_cuda(self, region: CudaSharedMemoryRegion, byte_size: int, offset: int) -> bytes:
        cudart = self._get_cudart()
        self._check_cuda(cudart.cudaSetDevice(region.device_id), "cudaSetDevice")
        if region.pointer is None:
            raise RuntimeError("CUDA shared memory pointer is not available")
        start = region.offset + offset
        end = start + byte_size
        if start < region.offset or end > region.offset + region.byte_size:
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
        if start < region.offset or end > region.offset + region.byte_size:
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
            raise ValueError("Requested byte_size must be positive")
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
_state_lock = asyncio.Lock()
_state_condition = asyncio.Condition(_state_lock)
_active_inference = 0
assets: Optional[Dict[str, Any]] = None


class AuthValidator:
    def __init__(self, tokens: Iterable[str], require: bool) -> None:
        self._tokens = {token for token in tokens if token}
        self._require = require

    def validate(self, authorization: Optional[str]) -> bool:
        if not self._require:
            return True
        if not self._tokens:
            return False
        if not authorization or not authorization.lower().startswith("bearer "):
            return False
        provided = authorization.split(" ", maxsplit=1)[1].strip()
        return provided in self._tokens

    def validate_grpc(self, context: grpc.ServicerContext) -> bool:
        metadata = {md.key.lower(): md.value for md in context.invocation_metadata()}
        return self.validate(metadata.get("authorization"))


class AsyncRateLimiter:
    def __init__(self, limit: Optional[int]) -> None:
        self._sem = asyncio.Semaphore(limit) if limit else None

    @asynccontextmanager
    async def limit(self):
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


class MetricsRecorder:
    def __init__(self, registry: CollectorRegistry):
        self.registry = registry
        self.requests = Counter(
            "nim_requests_total",
            "Total inference requests",
            ["protocol", "status"],
            registry=registry,
        )
        self.latency = Histogram(
            "nim_request_latency_seconds",
            "Request latency in seconds",
            ["protocol"],
            registry=registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self.batch_size = Histogram(
            "nim_request_batch_size",
            "Observed batch sizes",
            ["protocol"],
            registry=registry,
            buckets=tuple(float(size) for size in [1, 2, 4, 8, 16, 32, 64, 128]),
        )

    def observe(self, protocol: str, status: str, duration: float, batch_size: int) -> None:
        self.requests.labels(protocol=protocol, status=status).inc()
        self.latency.labels(protocol=protocol).observe(duration)
        self.batch_size.labels(protocol=protocol).observe(float(batch_size))


def _setup_tracer(settings: ServiceSettings) -> Optional["trace.Tracer"]:
    if not settings.enable_otel:
        return None
    if trace is None or OTLPSpanExporter is None or Resource is None or TracerProvider is None or BatchSpanProcessor is None:
        logger.warning("OpenTelemetry requested but exporters are unavailable; skipping OTEL setup")
        return None
    if settings.otel_traces_exporter.lower() != "otlp":
        return None
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


def _read_max_batch_size(settings: ServiceSettings) -> int:
    if settings.triton_config_path and os.path.exists(settings.triton_config_path):
        try:
            with open(settings.triton_config_path, "r", encoding="utf-8") as file:
                for line in file:
                    stripped = line.strip()
                    if stripped.startswith("max_batch_size"):
                        _, value = stripped.split(":", maxsplit=1)
                        parsed = int(value.strip())
                        if parsed > 0:
                            return parsed
        except Exception as exc:  # pragma: no cover - config parsing errors only logged
            logger.warning("Failed to parse Triton config at %s: %s", settings.triton_config_path, exc)
    if settings.max_batch_size_env is not None and settings.max_batch_size_env > 0:
        return settings.max_batch_size_env
    return 64


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_assets() -> Dict[str, Any]:
    device = _select_device()
    dtype = _select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()

    configured_max_length = (
        settings.max_sequence_length
        or getattr(model.config, "max_position_embeddings", None)
        or 8192
    )

    logger.info(
        "Loaded %s on %s with dtype %s (max_length=%d, batch_limit=%d)",
        settings.model_id,
        device,
        dtype,
        configured_max_length,
        triton_config.max_batch_size,
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "dtype": dtype,
        "max_length": int(configured_max_length),
    }


def _dispose_assets(current_assets: Optional[Dict[str, Any]]) -> None:
    if current_assets is None:
        return
    model = current_assets.get("model")
    try:
        if isinstance(model, torch.nn.Module):
            model.cpu()
    except Exception:  # pragma: no cover - defensive cleanup
        logger.debug("Failed to move model to CPU during unload", exc_info=True)
    current_assets.clear()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Failed to empty CUDA cache during unload", exc_info=True)


async def _record_inference_success(duration_ns: int) -> None:
    async with _state_lock:
        model_state.inference_count += 1
        model_state.success_ns_total += duration_ns
        model_state.last_inference_ns = time.time_ns()


async def _acquire_model_assets_http() -> Dict[str, Any]:
    global _active_inference
    async with _state_condition:
        if not model_state.loaded or assets is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        _active_inference += 1
        return assets


async def _acquire_model_assets_grpc(context: grpc.ServicerContext) -> Dict[str, Any]:
    global _active_inference
    async with _state_condition:
        if not model_state.loaded or assets is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        _active_inference += 1
        return assets


async def _release_inference_slot() -> None:
    global _active_inference
    async with _state_condition:
        if _active_inference > 0:
            _active_inference -= 1
        _state_condition.notify_all()


async def _load_model() -> None:
    global assets
    async with _state_condition:
        if model_state.loaded and assets is not None:
            return
        while _active_inference > 0:
            await _state_condition.wait()
    loaded_assets = await asyncio.to_thread(_load_assets)
    async with _state_condition:
        assets = loaded_assets
        model_state.loaded = True
        _state_condition.notify_all()


async def _unload_model() -> None:
    global assets
    async with _state_condition:
        if not model_state.loaded and assets is None:
            return
        while _active_inference > 0:
            await _state_condition.wait()
        current_assets = assets
        assets = None
        model_state.loaded = False
        _state_condition.notify_all()
    _dispose_assets(current_assets)


def _initial_model_load() -> None:
    global assets
    assets = _load_assets()
    model_state.loaded = True


def _format_pair(query_text: str, passage_text: str) -> str:
    return f"question:{query_text} \\n \\n passage:{passage_text}"


def _encode_pairs(
    pairs: List[str],
    truncate: TruncationMode,
    assets: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    tokenizer: AutoTokenizer = assets["tokenizer"]
    truncation_side = "left" if truncate == "START" else "right"
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side

    try:
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=truncate != "NONE",
            max_length=assets["max_length"],
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = previous_truncation_side

    return {key: value.to(assets["device"]) for key, value in encoded.items()}


def _score_passages(request_body: RankingRequest, assets: Dict[str, Any]) -> List[float]:
    pairs = [_format_pair(request_body.query.text, passage.text) for passage in request_body.passages]
    encoded = _encode_pairs(pairs, request_body.truncate, assets)
    model: AutoModelForSequenceClassification = assets["model"]

    with torch.inference_mode():
        logits = model(**encoded).logits

    return logits.view(-1).float().cpu().tolist()


def _current_model_ready() -> bool:
    return model_state.loaded and assets is not None


def _validate_model_name(model_name: Optional[str]) -> None:
    if model_name is None:
        return
    if model_name != settings.model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}', expected '{settings.model_id}'",
        )


def _validate_batch_size(request_body: RankingRequest) -> None:
    passage_count = len(request_body.passages)
    if passage_count > triton_config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {passage_count} exceeds limit {triton_config.max_batch_size}",
        )


def _parse_shared_memory_params(
    parameters: Mapping[str, service_pb2.ModelInferRequest.InferParameter],
) -> tuple[Optional[str], Optional[int], int]:
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


def _decode_tensor_payload(
    tensor: service_pb2.ModelInferRequest.InferInputTensor,
    raw_content: Optional[bytes],
) -> np.ndarray:
    shape = tuple(int(dim) for dim in tensor.shape)
    if tensor.datatype == "BYTES":
        if raw_content is not None:
            array = deserialize_bytes_tensor(raw_content)
        else:
            array = np.array(list(tensor.contents.bytes_contents), dtype=np.object_)
        if shape and np.prod(shape) == array.size:
            array = array.reshape(shape)
        return array

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
    if shape:
        array = array.reshape(shape)
    return array


def _resolve_output_region(
    outputs: List[service_pb2.ModelInferRequest.InferRequestedOutputTensor],
    request_parameters: Mapping[str, service_pb2.ModelInferRequest.InferParameter],
) -> tuple[Optional[str], Optional[int], int]:
    if outputs:
        region_name, region_size, region_offset = _parse_shared_memory_params(outputs[0].parameters)
        if region_name is not None:
            return region_name, region_size, region_offset
    return _parse_shared_memory_params(request_parameters)


async def _run_rank(request_body: RankingRequest) -> Dict[str, Any]:
    batch_size = len(request_body.passages)
    started = perf_counter()
    acquired_assets: Optional[Dict[str, Any]] = None
    try:
        _validate_model_name(request_body.model)
        _validate_batch_size(request_body)
        acquired_assets = await _acquire_model_assets_http()
        async with rate_limiter.limit():
            scores = await asyncio.to_thread(_score_passages, request_body, acquired_assets)
    except HTTPException:
        metrics.observe("http", "error", perf_counter() - started, batch_size)
        raise
    except ValueError as exc:
        metrics.observe("http", "error", perf_counter() - started, batch_size)
        logger.error("Invalid input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        metrics.observe("http", "error", perf_counter() - started, batch_size)
        logger.exception("Ranking failed: %s", exc)
        raise HTTPException(status_code=500, detail="Ranking failed") from exc
    finally:
        if acquired_assets is not None:
            await _release_inference_slot()

    rankings = sorted(
        [{"index": index, "logit": score} for index, score in enumerate(scores)],
        key=lambda item: item["logit"],
        reverse=True,
    )
    duration = perf_counter() - started
    metrics.observe("http", "success", duration, batch_size)
    await _record_inference_success(int(duration * 1e9))
    return {"rankings": rankings}


def _is_public_path(path: str) -> bool:
    if path == "/metrics":
        return not settings.require_auth
    return path in {
        "/",
        "/v1/health/live",
        "/v1/health/ready",
        "/v2/health/live",
        "/v2/health/ready",
    }


def _model_metadata_proto() -> service_pb2.ModelMetadataResponse:
    response = service_pb2.ModelMetadataResponse()
    response.name = triton_config.name
    response.platform = triton_config.platform
    response.versions.extend([triton_config.version])
    for spec in triton_config.inputs:
        tensor = response.inputs.add()
        tensor.name = spec.name
        tensor.datatype = spec.dtype
        tensor.shape.extend(spec.shape)
    for spec in triton_config.outputs:
        tensor = response.outputs.add()
        tensor.name = spec.name
        tensor.datatype = spec.dtype
        tensor.shape.extend(spec.shape)
    return response


def _model_config_proto() -> model_config_pb2.ModelConfig:
    cfg = model_config_pb2.ModelConfig(
        name=triton_config.name,
        platform=triton_config.platform,
        max_batch_size=triton_config.max_batch_size,
    )
    dtype_map = {"BYTES": model_config_pb2.TYPE_STRING, "FP32": model_config_pb2.TYPE_FP32}
    for spec in triton_config.inputs:
        cfg_input = cfg.input.add()
        cfg_input.name = spec.name
        cfg_input.data_type = dtype_map.get(spec.dtype, model_config_pb2.TYPE_STRING)
        cfg_input.dims.extend(spec.shape)
    for spec in triton_config.outputs:
        cfg_output = cfg.output.add()
        cfg_output.name = spec.name
        cfg_output.data_type = dtype_map.get(spec.dtype, model_config_pb2.TYPE_STRING)
        cfg_output.dims.extend(spec.shape)
    return cfg


class TritonRerankService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(
        self,
        config: TritonModelConfig,
        auth: AuthValidator,
        metrics_recorder: MetricsRecorder,
        state: ModelState,
        tracer_obj: Optional["trace.Tracer"],
    ) -> None:
        self._config = config
        self._auth = auth
        self._metrics = metrics_recorder
        self._state = state
        self._tracer = tracer_obj

    async def _ensure_auth(self, context: grpc.ServicerContext, operation: str) -> None:
        if self._auth.validate_grpc(context):
            return
        logger.warning("Unauthorized gRPC request for %s", operation)
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")

    @staticmethod
    def _decode_inputs(request: service_pb2.ModelInferRequest) -> Dict[str, np.ndarray]:
        decoded: Dict[str, np.ndarray] = {}
        raw_contents = list(request.raw_input_contents)
        raw_index = 0
        for tensor in request.inputs:
            raw_content = raw_contents[raw_index] if raw_index < len(raw_contents) else None
            if raw_content is not None:
                raw_index += 1
            try:
                region_name, region_size, region_offset = _parse_shared_memory_params(tensor.parameters)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            payload = None
            if region_name is not None:
                if region_size is None or region_size <= 0:
                    raise ValueError("shared_memory_byte_size must be positive when using shared memory")
                try:
                    payload = shared_memory_registry.read(region_name, int(region_size), int(region_offset))
                except KeyError as exc:
                    raise FileNotFoundError(str(exc)) from exc
            elif raw_content is not None:
                payload = raw_content
            decoded[tensor.name] = _decode_tensor_payload(tensor, payload)
        return decoded

    @staticmethod
    def _as_text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _extract_truncate(inputs: Dict[str, np.ndarray]) -> TruncationMode:
        if TRUNCATE_INPUT_NAME not in inputs:
            return "END"
        raw = inputs[TRUNCATE_INPUT_NAME]
        if raw.size == 0:
            return "END"
        candidate = TritonRerankService._as_text(raw.reshape(-1)[0]).strip().upper()
        if candidate in {"NONE", "END", "START"}:
            return candidate  # type: ignore[return-value]
        return "END"

    async def _run_infer_request(
        self,
        request: service_pb2.ModelInferRequest,
        context: grpc.aio.ServicerContext,
    ) -> service_pb2.ModelInferResponse:
        started = perf_counter()
        if not _current_model_ready():
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        if request.model_name and request.model_name not in {"", self._config.name}:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")

        try:
            decoded_inputs = self._decode_inputs(request)
        except FileNotFoundError as exc:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

        if QUERY_INPUT_NAME not in decoded_inputs or PASSAGES_INPUT_NAME not in decoded_inputs:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "QUERY and PASSAGES inputs are required")
        query_values = decoded_inputs[QUERY_INPUT_NAME].reshape(-1)
        passage_values = decoded_inputs[PASSAGES_INPUT_NAME].reshape(-1)
        if query_values.size == 0:
            self._metrics.observe("grpc", "error", perf_counter() - started, 0)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "QUERY input must include one string")
        passages = [self._as_text(value) for value in passage_values]
        truncate = self._extract_truncate(decoded_inputs)
        batch_size = len(passages)
        if batch_size == 0:
            self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "PASSAGES input must include at least one entry")
        if batch_size > self._config.max_batch_size:
            self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Batch size {batch_size} exceeds limit {self._config.max_batch_size}",
            )

        output_region, output_size, output_offset = _resolve_output_region(list(request.outputs), request.parameters)
        span_cm = self._tracer.start_as_current_span("grpc_rerank") if self._tracer is not None else nullcontext()
        assets_ref: Optional[Dict[str, Any]] = None
        with span_cm:
            try:
                request_body = RankingRequest.model_validate(
                    {
                        "query": {"text": self._as_text(query_values[0])},
                        "passages": [{"text": text} for text in passages],
                        "truncate": truncate,
                    }
                )
            except Exception as exc:  # pragma: no cover - validation errors surfaced to caller
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid input: {exc}")

            if self._tracer is not None:
                trace.get_current_span().set_attribute("rerank.batch_size", batch_size)
            try:
                assets_ref = await _acquire_model_assets_grpc(context)
                async with rate_limiter.limit():
                    scores = await asyncio.to_thread(_score_passages, request_body, assets_ref)
            except grpc.RpcError:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                raise
            except Exception as exc:  # pragma: no cover - surfaced to caller
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.INTERNAL, f"Inference failed: {exc}")
            finally:
                if assets_ref is not None:
                    await _release_inference_slot()

        response = service_pb2.ModelInferResponse()
        response.model_name = request.model_name or self._config.name
        response.model_version = self._config.version
        output = response.outputs.add()
        output.name = OUTPUT_NAME
        output.datatype = "FP32"
        output.shape.extend([len(scores)])
        serialized = np.asarray(scores, dtype=np.float32).tobytes()
        output_length = len(serialized)
        if output_region is not None:
            if output_size is None or output_size <= 0:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size must be positive for outputs")
            if output_offset < 0 or output_offset + output_length > output_size:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Shared memory region '{output_region}' is too small for output",
                )
            try:
                shared_memory_registry.write(output_region, serialized, int(output_offset))
                response.raw_output_contents[:] = []
            except KeyError:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Shared memory region '{output_region}' is not registered")
            except ValueError as exc:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                self._metrics.observe("grpc", "error", perf_counter() - started, batch_size)
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
        else:
            response.raw_output_contents.append(serialized)

        duration = perf_counter() - started
        self._metrics.observe("grpc", "success", duration, batch_size)
        await _record_inference_success(int(duration * 1e9))
        return response

    async def ModelInfer(self, request: service_pb2.ModelInferRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelInferResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelInfer")
        return await self._run_infer_request(request, context)

    async def ModelReady(self, request: service_pb2.ModelReadyRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelReadyResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelReady")
        ready = _current_model_ready() and (request.name in {"", self._config.name})
        return service_pb2.ModelReadyResponse(ready=ready)

    async def ServerReady(self, request: service_pb2.ServerReadyRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerReadyResponse:  # noqa: N802
        return service_pb2.ServerReadyResponse(ready=_current_model_ready())

    async def ServerLive(self, request: service_pb2.ServerLiveRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerLiveResponse:  # noqa: N802
        return service_pb2.ServerLiveResponse(live=True)

    async def ServerMetadata(self, request: service_pb2.ServerMetadataRequest, context: grpc.aio.ServicerContext) -> service_pb2.ServerMetadataResponse:  # noqa: N802
        await self._ensure_auth(context, "ServerMetadata")
        return service_pb2.ServerMetadataResponse(
            name=self._config.name,
            version=self._config.version,
            extensions=["model_repository", "metrics"],
        )

    async def ModelMetadata(self, request: service_pb2.ModelMetadataRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelMetadataResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelMetadata")
        if request.name and request.name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        return _model_metadata_proto()

    async def ModelConfig(self, request: service_pb2.ModelConfigRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelConfigResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelConfig")
        if request.name and request.name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        return model_config_pb2.ModelConfigResponse(config=_model_config_proto())

    async def ModelStatistics(self, request: service_pb2.ModelStatisticsRequest, context: grpc.aio.ServicerContext) -> service_pb2.ModelStatisticsResponse:  # noqa: N802
        await self._ensure_auth(context, "ModelStatistics")
        if request.name and request.name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.name}")
        async with _state_lock:
            inference_count = model_state.inference_count
            last_inference = model_state.last_inference_ns
            success_ns_total = model_state.success_ns_total
        stats = service_pb2.ModelStatistics(
            name=self._config.name,
            version=self._config.version,
            last_inference=last_inference,
            inference_count=inference_count,
            execution_count=inference_count,
        )
        stats.inference_stats.success.count = inference_count
        stats.inference_stats.success.ns = success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    async def RepositoryIndex(self, request: service_pb2.RepositoryIndexRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryIndexResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryIndex")
        entry = service_pb2.RepositoryIndexResponse.ModelIndex(
            name=self._config.name,
            version=self._config.version,
            state="READY" if _current_model_ready() else "UNAVAILABLE",
        )
        return service_pb2.RepositoryIndexResponse(models=[entry])

    async def RepositoryModelLoad(self, request: service_pb2.RepositoryModelLoadRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryModelLoadResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryModelLoad")
        if not settings.enable_model_control:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "explicit model load / unload is not allowed")
        if request.model_name and request.model_name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")
        try:
            await _load_model()
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to load model on demand: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to load model")
        return service_pb2.RepositoryModelLoadResponse()

    async def RepositoryModelUnload(self, request: service_pb2.RepositoryModelUnloadRequest, context: grpc.aio.ServicerContext) -> service_pb2.RepositoryModelUnloadResponse:  # noqa: N802
        await self._ensure_auth(context, "RepositoryModelUnload")
        if not settings.enable_model_control:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "explicit model load / unload is not allowed")
        if request.model_name and request.model_name != self._config.name:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model {request.model_name}")
        try:
            await _unload_model()
        except Exception as exc:  # pragma: no cover - surfaced to caller
            logger.exception("Failed to unload model on demand: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to unload model")
        return service_pb2.RepositoryModelUnloadResponse()

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

    async def TraceSetting(self, request, context):  # noqa: N802
        return service_pb2.TraceSettingResponse()

    async def LogSettings(self, request, context):  # noqa: N802
        return service_pb2.LogSettingsResponse()

    async def CudaSharedMemoryRegister(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryRegister")
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
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to register CUDA shared memory: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to register CUDA shared memory")

    async def CudaSharedMemoryStatus(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryStatus")
        response = service_pb2.CudaSharedMemoryStatusResponse()
        regions = shared_memory_registry.cuda_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' is not registered")
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.device_id = region.device_id
            status.byte_size = region.byte_size
        return response

    async def CudaSharedMemoryUnregister(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "CudaSharedMemoryUnregister")
        if not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_cuda(request.name)
            return service_pb2.CudaSharedMemoryUnregisterResponse()
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to unregister CUDA shared memory: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to unregister CUDA shared memory")

    async def SystemSharedMemoryRegister(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryRegister")
        try:
            shared_memory_registry.register_system(
                request.name,
                request.key,
                int(request.offset),
                int(request.byte_size),
            )
            return service_pb2.SystemSharedMemoryRegisterResponse()
        except FileExistsError as exc:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to register system shared memory: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to register system shared memory")

    async def SystemSharedMemoryStatus(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryStatus")
        response = service_pb2.SystemSharedMemoryStatusResponse()
        regions = shared_memory_registry.system_regions()
        if request.name:
            regions = [region for region in regions if region.name == request.name]
            if not regions:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"System shared memory region '{request.name}' is not registered",
                )
        for region in regions:
            status = response.regions[region.name]
            status.name = region.name
            status.key = region.key
            status.offset = region.offset
            status.byte_size = region.byte_size
        return response

    async def SystemSharedMemoryUnregister(self, request, context):  # noqa: N802
        await self._ensure_auth(context, "SystemSharedMemoryUnregister")
        if not request.name:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            shared_memory_registry.unregister_system(request.name)
            return service_pb2.SystemSharedMemoryUnregisterResponse()
        except FileNotFoundError as exc:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to unregister system shared memory: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to unregister system shared memory")


settings = ServiceSettings()
logging.getLogger().setLevel(settings.log_level.upper())
if settings.log_verbose > 0 and logging.getLogger().level > logging.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)

max_batch_size = _read_max_batch_size(settings)
triton_config = TritonModelConfig(
    name=settings.model_id,
    version=settings.model_version,
    platform="python",
    max_batch_size=max_batch_size,
    inputs=[
        TensorSpec(name=QUERY_INPUT_NAME, dtype="BYTES", shape=[1]),
        TensorSpec(name=PASSAGES_INPUT_NAME, dtype="BYTES", shape=[-1]),
        TensorSpec(name=TRUNCATE_INPUT_NAME, dtype="BYTES", shape=[1]),
    ],
    outputs=[TensorSpec(name=OUTPUT_NAME, dtype="FP32", shape=[-1])],
)

tracer = _setup_tracer(settings)
auth_validator = AuthValidator(settings.resolved_auth_tokens, settings.require_auth)
metrics_registry = CollectorRegistry()
metrics = MetricsRecorder(metrics_registry)
rate_limiter = AsyncRateLimiter(settings.rate_limit if settings.rate_limit is not None and settings.rate_limit > 0 else None)
model_state = ModelState()
_initial_model_load()

grpc_service = TritonRerankService(triton_config, auth_validator, metrics, model_state, tracer)
app = FastAPI(title=settings.model_id, version=settings.model_version)
_metrics_started = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _metrics_started
    if not _metrics_started and not settings.require_auth:
        start_http_server(settings.metrics_port, registry=metrics_registry)
        _metrics_started = True
    server = grpc.aio.server()
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(grpc_service, server)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    await server.start()
    app.state.grpc_server = server
    logger.info("Started Triton-compatible gRPC server on :%d", settings.grpc_port)
    try:
        yield
    finally:
        await _unload_model()
        await server.stop(grace=None)


app.router.lifespan_context = lifespan


@app.middleware("http")
async def _log_and_auth(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    logger.info("%s %s", request.method, request.url.path)
    span_cm = tracer.start_as_current_span(f"http {request.method.lower()} {request.url.path}") if tracer else nullcontext()
    with span_cm:
        if tracer:
            trace.get_current_span().set_attribute("http.method", request.method)
            trace.get_current_span().set_attribute("http.route", request.url.path)
        if not _is_public_path(request.url.path):
            if not auth_validator.validate(request.headers.get("authorization")):
                if tracer:
                    trace.get_current_span().set_attribute("http.status_code", 401)
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        response = await call_next(request)
        if tracer:
            trace.get_current_span().set_attribute("http.status_code", response.status_code)
        return response


@app.get("/v1/health/live")
async def live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v1/health/ready")
async def ready() -> JSONResponse:
    is_ready = _current_model_ready()
    status_value = 200 if is_ready else 503
    return JSONResponse({"ready": is_ready}, status_code=status_value)


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": settings.model_id}]}


@app.get("/v1/metadata")
async def metadata() -> Dict[str, Any]:
    short_name = f"{settings.model_id}:{settings.model_version}"
    return {
        "id": settings.model_id,
        "name": settings.model_id,
        "version": settings.model_version,
        "modelInfo": [
            {
                "name": settings.model_id,
                "version": settings.model_version,
                "shortName": short_name,
                "maxBatchSize": triton_config.max_batch_size,
            }
        ],
    }


@app.get("/metrics")
async def metrics_endpoint() -> PlainTextResponse:
    content = generate_latest(metrics_registry)
    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")


@app.post("/v1/ranking")
async def ranking(request_body: RankingRequest) -> Dict[str, Any]:
    return await _run_rank(request_body)


@app.post("/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking")
async def compatibility_ranking(request_body: RankingRequest) -> Dict[str, Any]:
    return await _run_rank(request_body)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"message": settings.model_id}, status_code=200)


@app.get("/v2/health/live")
async def triton_live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready")
async def triton_ready() -> Dict[str, bool]:
    return {"ready": _current_model_ready()}


@app.get("/v2/models")
async def list_triton_models() -> Dict[str, object]:
    state = "READY" if _current_model_ready() else "UNAVAILABLE"
    return {"models": [{"name": triton_config.name, "version": triton_config.version, "state": state}]}


@app.get("/v2/models/{model_name}")
async def model_metadata_http(model_name: str) -> Dict[str, object]:
    if model_name != triton_config.name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    proto = _model_metadata_proto()
    return json_format.MessageToDict(proto, preserving_proto_field_name=True)


@app.get("/v2/models/{model_name}/config")
async def model_config_http(model_name: str) -> Dict[str, object]:
    if model_name != triton_config.name:
        raise HTTPException(status_code=404, detail=f"Unknown model {model_name}")
    proto = _model_config_proto()
    return json_format.MessageToDict(proto, preserving_proto_field_name=True)


@app.get("/v2/models/{model_name}/ready")
async def model_ready_http(model_name: str) -> Dict[str, bool]:
    is_ready = model_name == triton_config.name and _current_model_ready()
    return {"ready": is_ready}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.http_port)
