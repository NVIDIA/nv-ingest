from __future__ import annotations

import io
import json
import logging
import signal
import threading
import time
from collections import deque
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import grpc
import numpy as np
import torch
from google.protobuf.json_format import MessageToDict
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from scipy.io import wavfile
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc

from riva.client.proto import riva_asr_pb2 as rasr
from riva.client.proto import riva_asr_pb2_grpc as rasr_grpc


LOGGER = logging.getLogger(__name__)


@dataclass
class EndpointingOptions:
    start_history_seconds: float
    start_threshold: float
    stop_history_seconds: float
    stop_threshold: float
    eou_history_seconds: float
    eou_threshold: float

    @property
    def max_history_seconds(self) -> float:
        return max(
            self.start_history_seconds,
            self.stop_history_seconds,
            self.eou_history_seconds,
        )


@dataclass
class RecognitionOptions:
    sample_rate: int
    language_code: str
    include_word_time_offsets: bool
    enable_punctuation: bool
    enable_diarization: bool
    interim_results: bool
    channel_count: int
    max_speaker_count: int
    endpointing: EndpointingOptions


@dataclass
class StreamingState:
    options: RecognitionOptions
    audio_chunks: List[np.ndarray] = field(default_factory=list)
    logits: List[np.ndarray] = field(default_factory=list)
    total_audio_seconds: float = 0.0
    last_emit_time: float = 0.0
    last_transcript: Optional[str] = None
    vad_history: Deque[Tuple[float, float]] = field(default_factory=deque)
    speech_started: bool = False
    endpoint_triggered: bool = False
    last_silence_time: Optional[float] = None
    final_emitted: bool = False

    def append(self, audio_chunk: np.ndarray, new_logits: np.ndarray, duration_seconds: float) -> None:
        self.audio_chunks.append(audio_chunk)
        self.logits.append(new_logits)
        self.total_audio_seconds += duration_seconds

    def merged_logits(self) -> Optional[np.ndarray]:
        if not self.logits:
            return None
        return np.concatenate(self.logits, axis=0)

    def merged_audio(self) -> Optional[np.ndarray]:
        if not self.audio_chunks:
            return None
        return np.concatenate(self.audio_chunks)


class ServiceSettings(BaseSettings):
    model_id: str = Field("nvidia/parakeet-ctc-1.1b", alias="MODEL_ID")
    model_name: str = Field("parakeet-1-1b-ctc-en-us", alias="MODEL_NAME")
    model_version: str = Field("1.1.0", alias="MODEL_VERSION")
    grpc_port: int = Field(50051, alias="GRPC_PORT")
    http_port: int = Field(9000, alias="HTTP_PORT")
    max_workers: int = Field(4, alias="MAX_WORKERS")
    max_concurrent_requests: int = Field(16, alias="MAX_CONCURRENT_REQUESTS")
    max_streaming_sessions: int = Field(8, alias="MAX_STREAMING_SESSIONS")
    max_batch_size: int = Field(4, alias="MAX_BATCH_SIZE")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    auth_bearer_tokens: List[str] = Field(default_factory=list, alias="AUTH_TOKENS")
    auth_ngc_api_keys: List[str] = Field(default_factory=list, alias="NGC_API_KEYS")
    require_auth: bool = Field(False, alias="REQUIRE_AUTH")
    allow_unauth_health: bool = Field(True, alias="ALLOW_UNAUTH_HEALTH")
    default_language_code: str = Field("en-US", alias="LANGUAGE_CODE")
    default_punctuation: bool = Field(False, alias="ENABLE_AUTOMATIC_PUNCTUATION")
    punctuation_model_id: str = Field("kredor/punctuate-all", alias="PUNCTUATION_MODEL_ID")
    default_diarization: bool = Field(False, alias="ENABLE_DIARIZATION")
    diarization_max_speakers: int = Field(2, alias="MAX_SPEAKER_COUNT")
    streaming_emit_interval_ms: int = Field(400, alias="STREAMING_EMIT_INTERVAL_MS")
    endpoint_start_history_ms: int = Field(240, alias="ENDPOINT_START_HISTORY_MS")
    endpoint_start_threshold: float = Field(0.6, alias="ENDPOINT_START_THRESHOLD")
    endpoint_stop_history_ms: int = Field(1200, alias="ENDPOINT_STOP_HISTORY_MS")
    endpoint_stop_threshold: float = Field(0.35, alias="ENDPOINT_STOP_THRESHOLD")
    endpoint_eou_history_ms: int = Field(1600, alias="ENDPOINT_EOU_HISTORY_MS")
    endpoint_eou_threshold: float = Field(0.2, alias="ENDPOINT_EOU_THRESHOLD")
    vad_frame_ms: int = Field(30, alias="VAD_FRAME_MS")
    metrics_namespace: str = Field("parakeet", alias="METRICS_NAMESPACE")
    otel_endpoint: Optional[str] = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str = Field("parakeet-asr", alias="OTEL_SERVICE_NAME")

    @field_validator("auth_bearer_tokens", "auth_ngc_api_keys", mode="before")
    @classmethod
    def _split_tokens(cls, value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("token lists must be provided as comma-separated strings or lists")

    class Config:
        extra = "ignore"


@dataclass
class TritonModelConfig:
    name: str
    version: str
    max_batch_size: int
    sample_rate: int
    language_code: str


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _setup_tracer(settings: ServiceSettings) -> Optional[trace.Tracer]:
    if not settings.otel_endpoint:
        return None

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.model_version,
            "service.instance.id": settings.model_name,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(settings.otel_service_name)


class Metrics:
    def __init__(self, namespace: str):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "requests_total",
            "Total requests by method and status",
            ["method", "status"],
            namespace=namespace,
            registry=self.registry,
        )
        self.latency = Histogram(
            "request_duration_seconds",
            "Latency per method",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )
        self.active = Gauge(
            "active_requests",
            "In-flight requests by method",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )
        self.audio_seconds = Counter(
            "audio_processed_seconds_total",
            "Total audio seconds processed",
            ["method"],
            namespace=namespace,
            registry=self.registry,
        )

    @contextmanager
    def track(self, method: str):
        start = time.perf_counter()
        self.active.labels(method=method).inc()
        status = "ok"
        exc: Optional[BaseException] = None
        try:
            yield
        except Exception as exc_:  # pragma: no cover - defensive guard
            status = "error"
            exc = exc_
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.latency.labels(method=method).observe(elapsed)
            self.requests.labels(method=method, status=status).inc()
            self.active.labels(method=method).dec()
            if exc:
                LOGGER.debug("request %s failed: %s", method, exc)


class BusyError(RuntimeError):
    pass


class RequestLimiter:
    def __init__(self, max_concurrent_requests: int, max_streaming_sessions: int):
        self.general = threading.BoundedSemaphore(value=max(1, max_concurrent_requests))
        self.streaming = threading.BoundedSemaphore(value=max(1, max_streaming_sessions))

    @contextmanager
    def limit(self, semaphore: threading.Semaphore):
        acquired = semaphore.acquire(blocking=False)
        if not acquired:
            raise BusyError("server is busy")
        try:
            yield
        finally:
            semaphore.release()


class AuthValidator:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.enabled = settings.require_auth or bool(settings.auth_bearer_tokens or settings.auth_ngc_api_keys)
        self.allowed_tokens = {token for token in settings.auth_bearer_tokens}
        self.allowed_ngc = {token for token in settings.auth_ngc_api_keys}

    def _extract_token(self, metadata: Dict[str, str]) -> Optional[str]:
        header = metadata.get("authorization") or metadata.get("grpcgateway-authorization")
        if header:
            if header.lower().startswith("bearer "):
                return header.split(" ", 1)[1].strip()
            return header.strip()
        ngc = metadata.get("ngc-api-key") or metadata.get("ngc-token")
        if ngc:
            return ngc.strip()
        return None

    def validate(self, metadata: Dict[str, str], allow_unauthenticated: bool = False) -> None:
        if not self.enabled or allow_unauthenticated:
            return
        token = self._extract_token(metadata)
        if token and (token in self.allowed_tokens or token in self.allowed_ngc):
            return
        raise PermissionError("authentication failed")


class VoiceActivityDetector:
    def __init__(self, frame_ms: int):
        self.frame_ms = max(10, min(frame_ms, 30))
        self._vad = None
        try:
            import webrtcvad  # type: ignore
            self._vad = webrtcvad.Vad(2)
        except Exception:
            self._vad = None

    def _frame_audio(self, audio: np.ndarray, frame_length: int) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        if frame_length <= 0:
            return frames
        for start in range(0, audio.shape[0], frame_length):
            end = min(start + frame_length, audio.shape[0])
            frame = audio[start:end]
            if frame.size < frame_length:
                pad = np.zeros(frame_length - frame.size, dtype=audio.dtype)
                frame = np.concatenate([frame, pad])
            frames.append(frame)
        return frames

    def _to_pcm(self, frame: np.ndarray) -> bytes:
        clipped = np.clip(frame, -1.0, 1.0)
        return (clipped * 32768.0).astype(np.int16).tobytes()

    def _energy_probability(self, energy: float, reference: float) -> float:
        if reference <= 0.0:
            return 0.0
        scaled = energy / (reference * 4.0)
        return float(max(0.0, min(1.0, scaled)))

    def probabilities(self, audio: np.ndarray, sample_rate: int) -> Tuple[List[float], float]:
        if audio.size == 0 or sample_rate <= 0:
            return [], 0.0
        frame_length = int(sample_rate * (self.frame_ms / 1000.0))
        frames = self._frame_audio(audio, frame_length)
        if not frames:
            return [], 0.0
        energies = [float(np.mean(np.square(frame))) for frame in frames]
        reference_energy = max(float(np.median(energies)), 1e-8)
        probabilities: List[float] = []
        for frame, energy in zip(frames, energies):
            vad_score = None
            if self._vad is not None:
                try:
                    speech = self._vad.is_speech(self._to_pcm(frame), sample_rate)
                    vad_score = 1.0 if speech else 0.0
                except Exception:
                    vad_score = None
            energy_score = self._energy_probability(energy, reference_energy)
            if vad_score is None:
                probabilities.append(energy_score * 0.8)
            else:
                probabilities.append((0.9 * vad_score) + (0.1 * energy_score))
        frame_duration = float(frame_length) / float(sample_rate)
        return probabilities, frame_duration


def _decode_wav(audio_bytes: bytes, target_sample_rate: int) -> Tuple[np.ndarray, float]:
    if not audio_bytes:
        raise ValueError("audio payload is empty")
    try:
        source_rate, data = wavfile.read(io.BytesIO(audio_bytes))
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("unable to parse WAV payload") from exc
    if data.size == 0:
        raise ValueError("audio payload contained no samples")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in {"i", "u"}:
        max_range = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(max_range)
    elif data.dtype.kind == "f":
        data = data.astype(np.float32)
    else:
        raise ValueError(f"unsupported audio dtype {data.dtype}")
    if source_rate != target_sample_rate:
        data = resample_poly(data, target_sample_rate, source_rate)
    duration_seconds = float(data.shape[0]) / float(target_sample_rate)
    return data, duration_seconds


def _decode_pcm(
    audio_bytes: bytes,
    source_sample_rate: int,
    target_sample_rate: int,
    channel_count: int,
) -> Tuple[np.ndarray, float]:
    if not audio_bytes:
        raise ValueError("audio payload is empty")
    if source_sample_rate <= 0:
        raise ValueError("sample_rate_hertz must be set for PCM audio")
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    if pcm.size == 0:
        raise ValueError("audio payload contained no samples")
    if channel_count > 1:
        pcm = pcm.reshape(-1, channel_count).mean(axis=1)
    data = pcm.astype(np.float32) / 32768.0
    if source_sample_rate != target_sample_rate:
        data = resample_poly(data, target_sample_rate, source_sample_rate)
    duration_seconds = float(data.shape[0]) / float(target_sample_rate)
    return data, duration_seconds


def _endpointing_from_config(config: rasr.RecognitionConfig, settings: ServiceSettings) -> EndpointingOptions:
    cfg = config.endpointing_config if config.HasField("endpointing_config") else None

    def _int_field(value: object, default: int) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return default

    def _float_field(value: object, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    start_history_ms = _int_field(getattr(cfg, "start_history", settings.endpoint_start_history_ms), settings.endpoint_start_history_ms)
    start_threshold = _float_field(getattr(cfg, "start_threshold", settings.endpoint_start_threshold), settings.endpoint_start_threshold)
    stop_history_ms = _int_field(getattr(cfg, "stop_history", settings.endpoint_stop_history_ms), settings.endpoint_stop_history_ms)
    stop_threshold = _float_field(getattr(cfg, "stop_threshold", settings.endpoint_stop_threshold), settings.endpoint_stop_threshold)
    eou_history_ms = _int_field(getattr(cfg, "stop_history_eou", settings.endpoint_eou_history_ms), settings.endpoint_eou_history_ms)
    eou_threshold = _float_field(getattr(cfg, "stop_threshold_eou", settings.endpoint_eou_threshold), settings.endpoint_eou_threshold)

    return EndpointingOptions(
        start_history_seconds=float(start_history_ms) / 1000.0,
        start_threshold=start_threshold,
        stop_history_seconds=float(stop_history_ms) / 1000.0,
        stop_threshold=stop_threshold,
        eou_history_seconds=float(eou_history_ms) / 1000.0,
        eou_threshold=eou_threshold,
    )


def _trim_audio_with_vad(
    audio: np.ndarray,
    sample_rate: int,
    detector: VoiceActivityDetector,
    options: EndpointingOptions,
) -> Tuple[np.ndarray, float]:
    probabilities, frame_duration = detector.probabilities(audio, sample_rate)
    if not probabilities or frame_duration <= 0.0:
        return audio, 0.0
    start_window = max(1, int(round(options.start_history_seconds / frame_duration)))
    stop_window = max(1, int(round(options.stop_history_seconds / frame_duration)))

    def _rolling_average(values: Sequence[float], index: int, window: int) -> float:
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        if not window_values:
            return 0.0
        return float(sum(window_values)) / float(len(window_values))

    start_index = 0
    for idx in range(len(probabilities)):
        if _rolling_average(probabilities, idx, start_window) >= options.start_threshold:
            start_index = idx
            break
    end_index = len(probabilities) - 1
    for idx in range(len(probabilities) - 1, -1, -1):
        if _rolling_average(probabilities, idx, stop_window) >= options.stop_threshold:
            end_index = idx
            break
    start_sample = int(start_index * frame_duration * float(sample_rate))
    end_sample = min(audio.shape[0], int((end_index + 1) * frame_duration * float(sample_rate)))
    if end_sample <= start_sample:
        return audio, 0.0
    trimmed = audio[start_sample:end_sample]
    offset_seconds = float(start_sample) / float(sample_rate)
    return trimmed, offset_seconds


def _extract_offsets(raw_offsets: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    cleaned: List[Dict[str, float]] = []
    for item in raw_offsets:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        if word is None or start_offset is None or end_offset is None:
            continue
        cleaned.append(
            {
                "word": str(word),
                "start_offset": float(start_offset),
                "end_offset": float(end_offset),
            }
        )
    return cleaned


def _update_endpointing_state(
    state: StreamingState,
    options: EndpointingOptions,
    probabilities: Sequence[float],
    frame_duration: float,
) -> None:
    if not probabilities or frame_duration <= 0.0:
        return
    start_time = max(0.0, state.total_audio_seconds - (len(probabilities) * frame_duration))
    current_time = start_time
    for prob in probabilities:
        current_time += frame_duration
        state.vad_history.append((current_time, float(prob)))
        while state.vad_history and current_time - state.vad_history[0][0] > options.max_history_seconds:
            state.vad_history.popleft()
    if not state.vad_history:
        return

    def _average(window_seconds: float) -> float:
        if window_seconds <= 0.0:
            return 0.0
        cutoff = state.vad_history[-1][0] - window_seconds
        values = [prob for timestamp, prob in state.vad_history if timestamp >= cutoff]
        if not values:
            return 0.0
        return float(sum(values)) / float(len(values))

    start_average = _average(options.start_history_seconds)
    if not state.speech_started and start_average >= options.start_threshold:
        state.speech_started = True
    if not state.speech_started:
        return
    stop_average = _average(options.stop_history_seconds)
    eou_average = _average(options.eou_history_seconds)
    if stop_average <= options.stop_threshold:
        state.last_silence_time = state.vad_history[-1][0]
    if (
        not state.endpoint_triggered
        and state.last_silence_time is not None
        and (state.vad_history[-1][0] - state.last_silence_time) >= options.eou_history_seconds
        and eou_average <= options.eou_threshold
    ):
        state.endpoint_triggered = True


def _pipeline_vad_probabilities(state: StreamingState, max_frames: int = 64) -> List[float]:
    if not state.vad_history:
        return []
    history = list(state.vad_history)
    if max_frames > 0:
        history = history[-max_frames:]
    return [prob for _, prob in history]


class PunctuationRestorer:
    def __init__(self, model_id: str):
        try:
            from deepmultilingualpunctuation import PunctuationModel
        except Exception as exc:  # pragma: no cover - defensive import
            raise RuntimeError(
                "Punctuation model is required for punctuation restoration. Install deepmultilingualpunctuation."
            ) from exc
        self.model = PunctuationModel(model_id)

    def restore(self, transcript: str) -> str:
        cleaned = transcript.strip()
        if not cleaned:
            return transcript
        return self.model.restore_punctuation(cleaned)


def _apply_punctuation(transcript: str, restorer: Optional[PunctuationRestorer]) -> str:
    cleaned = transcript.strip()
    if not cleaned:
        return transcript
    if restorer is None:
        punctuated = cleaned[0].upper() + cleaned[1:]
        if punctuated[-1] not in {".", "!", "?"}:
            punctuated = f"{punctuated}."
        return punctuated
    try:
        return restorer.restore(cleaned)
    except Exception:  # pragma: no cover - best-effort fallback
        LOGGER.warning("Punctuation model failed; falling back to heuristic punctuation", exc_info=True)
        punctuated = cleaned[0].upper() + cleaned[1:]
        if punctuated[-1] not in {".", "!", "?"}:
            punctuated = f"{punctuated}."
        return punctuated


def _spectral_features(audio: np.ndarray, sample_rate: int, band_count: int = 8) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / float(sample_rate)) if audio.size else np.array([])
    band_edges = np.linspace(0.0, freqs[-1] if freqs.size else 0.0, band_count + 1)
    features: List[float] = []
    for idx in range(band_count):
        mask = (freqs >= band_edges[idx]) & (freqs < band_edges[idx + 1])
        if not np.any(mask):
            features.append(-12.0)
            continue
        energy = float(np.mean(np.square(spectrum[mask])))
        features.append(np.log(energy + 1e-8))
    zero_crossings = float(np.mean(np.abs(np.diff(np.sign(audio))))) if audio.size else 0.0
    features.append(zero_crossings)
    return np.array(features, dtype=np.float32)


def _kmeans(features: np.ndarray, clusters: int, iterations: int = 24) -> np.ndarray:
    centroids = features[:clusters].copy()
    labels = np.zeros(features.shape[0], dtype=np.int32)
    for _ in range(iterations):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for idx in range(clusters):
            mask = labels == idx
            if np.any(mask):
                centroids[idx] = features[mask].mean(axis=0)
    return labels


def _estimate_speaker_segments(
    audio: np.ndarray,
    sample_rate: int,
    max_speakers: int,
) -> List[Tuple[float, float, int]]:
    if audio.size == 0 or sample_rate <= 0 or max_speakers <= 0:
        return []
    window_size = int(sample_rate * 1.2)
    hop_size = int(sample_rate * 0.6)
    if window_size <= 0 or hop_size <= 0:
        return []
    energies: List[float] = []
    features: List[np.ndarray] = []
    timestamps: List[Tuple[float, float]] = []
    for start in range(0, max(audio.shape[0] - window_size + 1, 1), hop_size):
        end = min(start + window_size, audio.shape[0])
        window_audio = audio[start:end]
        energy = float(np.mean(np.square(window_audio)))
        energies.append(energy)
        timestamps.append((start / float(sample_rate), end / float(sample_rate)))
        features.append(_spectral_features(window_audio, sample_rate))
    if not energies:
        return []
    threshold = max(np.median(energies) * 0.5, 1e-6)
    voiced_indices = [idx for idx, energy in enumerate(energies) if energy >= threshold]
    if not voiced_indices:
        return []
    voiced_features = np.stack([features[idx] for idx in voiced_indices], axis=0)
    cluster_count = min(max_speakers, voiced_features.shape[0])
    labels = np.zeros(len(voiced_indices), dtype=np.int32)
    if cluster_count > 1:
        labels = _kmeans(voiced_features, cluster_count)
    segments: List[Tuple[float, float, int]] = []
    for idx, frame_index in enumerate(voiced_indices):
        start, end = timestamps[frame_index]
        speaker = int(labels[idx])
        if segments and segments[-1][2] == speaker and start - segments[-1][1] <= 0.25:
            last_start, _, last_speaker = segments[-1]
            segments[-1] = (last_start, end, last_speaker)
        else:
            segments.append((start, end, speaker))
    return segments


def _assign_speakers(
    raw_offsets: List[Dict[str, float]],
    segments: List[Tuple[float, float, int]],
    time_offset_seconds: float,
) -> Dict[int, int]:
    if not segments or not raw_offsets:
        return {}
    assignments: Dict[int, int] = {}
    for idx, entry in enumerate(raw_offsets):
        midpoint = (entry["start_offset"] + entry["end_offset"]) * 0.5 * time_offset_seconds
        speaker = 0
        for start, end, label in segments:
            if start <= midpoint <= end:
                speaker = label + 1
                break
        if speaker == 0:
            closest = min(segments, key=lambda seg: abs(midpoint - (seg[0] + seg[1]) * 0.5))
            speaker = closest[2] + 1
        assignments[idx] = speaker
    return assignments


class ASRBundle:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.device = _select_device()
        self.dtype = _select_dtype(self.device)
        self.processor = AutoProcessor.from_pretrained(settings.model_id)
        self.model = AutoModelForCTC.from_pretrained(
            settings.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.vad_detector = VoiceActivityDetector(settings.vad_frame_ms)
        self.punctuation_restorer: Optional[PunctuationRestorer] = None

        sampling_rate = getattr(self.processor.feature_extractor, "sampling_rate", 16000)
        self.sample_rate = int(sampling_rate)
        logits_ratio = getattr(self.model.config, "inputs_to_logits_ratio", 1.0)
        self.time_offset_seconds = float(logits_ratio) / float(self.sample_rate)

        LOGGER.info(
            "Loaded %s on %s (dtype=%s, sample_rate=%d, time_offset=%.6f)",
            settings.model_id,
            self.device,
            self.dtype,
            self.sample_rate,
            self.time_offset_seconds,
        )

    def _ensure_punctuation_restorer(self) -> Optional[PunctuationRestorer]:
        if self.punctuation_restorer is None:
            self.punctuation_restorer = PunctuationRestorer(self.settings.punctuation_model_id)
        return self.punctuation_restorer

    def _logits_from_audio(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="longest",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        return logits.detach().to(torch.float32).cpu().numpy()[0]

    def _decode_logits(
        self,
        logits: np.ndarray,
        include_word_offsets: bool,
    ) -> Tuple[str, List[Dict[str, float]]]:
        predicted_ids = np.argmax(logits, axis=-1)
        if predicted_ids.ndim == 1:
            predicted_ids = np.expand_dims(predicted_ids, axis=0)
        decoded = self.processor.batch_decode(
            predicted_ids,
            output_word_offsets=include_word_offsets,
        )
        transcript = decoded.text[0] if hasattr(decoded, "text") else decoded[0]
        if include_word_offsets and hasattr(decoded, "word_offsets"):
            offsets_source = decoded.word_offsets[0] if decoded.word_offsets else []
            raw_offsets = _extract_offsets(offsets_source)
        else:
            raw_offsets = []
        return transcript, raw_offsets

    def _run_model(
        self,
        audio: np.ndarray,
        include_word_offsets: bool,
    ) -> Tuple[str, List[Dict[str, float]]]:
        logits = self._logits_from_audio(audio)
        return self._decode_logits(logits, include_word_offsets)

    def _word_infos(
        self,
        raw_offsets: List[Dict[str, float]],
        speaker_assignments: Optional[Dict[int, int]] = None,
        offset_ms: float = 0.0,
    ) -> List[rasr.WordInfo]:
        words: List[rasr.WordInfo] = []
        for idx, entry in enumerate(raw_offsets):
            word_info = rasr.WordInfo()
            word_info.word = entry["word"]
            base_offset_ms = max(0.0, offset_ms)
            word_info.start_time = int(entry["start_offset"] * 1000 * self.time_offset_seconds + base_offset_ms)
            word_info.end_time = int(entry["end_offset"] * 1000 * self.time_offset_seconds + base_offset_ms)
            if speaker_assignments:
                word_info.speaker_tag = int(speaker_assignments.get(idx, 0))
            words.append(word_info)
        return words

    def recognize_from_audio(
        self,
        audio: np.ndarray,
        include_word_offsets: bool,
        enable_diarization: bool,
        enable_punctuation: bool,
        max_speaker_count: int,
        offset_seconds: float = 0.0,
        original_duration_seconds: Optional[float] = None,
    ) -> rasr.SpeechRecognitionResult:
        logits = self._logits_from_audio(audio)
        transcript, offsets = self._decode_logits(logits, include_word_offsets)
        if enable_punctuation:
            transcript = _apply_punctuation(transcript, self._ensure_punctuation_restorer())
        alternative = rasr.SpeechRecognitionAlternative()
        alternative.transcript = transcript
        alternative.confidence = 0.0
        speaker_assignments: Optional[Dict[int, int]] = None
        if enable_diarization and include_word_offsets:
            segments = _estimate_speaker_segments(audio, self.sample_rate, max(1, max_speaker_count))
            speaker_assignments = _assign_speakers(offsets, segments, self.time_offset_seconds)
        if include_word_offsets:
            alternative.words.extend(
                self._word_infos(
                    offsets,
                    speaker_assignments,
                    offset_ms=max(0.0, offset_seconds) * 1000.0,
                )
            )
        result = rasr.SpeechRecognitionResult()
        result.alternatives.append(alternative)
        processed_seconds = float(audio.shape[0]) / float(self.sample_rate)
        result.audio_processed = float(original_duration_seconds) if original_duration_seconds is not None else processed_seconds + max(0.0, offset_seconds)
        return result

    def decode_bytes(
        self,
        payload: bytes,
        config: rasr.RecognitionConfig,
    ) -> Tuple[np.ndarray, float]:
        encoding = config.encoding
        if encoding == rasr.RecognitionConfig.AudioEncoding.LINEAR_PCM:
            return _decode_pcm(
                payload,
                config.sample_rate_hertz or self.sample_rate,
                self.sample_rate,
                max(1, config.audio_channel_count or 1),
            )
        try:
            return _decode_wav(payload, self.sample_rate)
        except ValueError:
            # fallback to PCM if WAV parsing fails
            return _decode_pcm(
                payload,
                config.sample_rate_hertz or self.sample_rate,
                self.sample_rate,
                max(1, config.audio_channel_count or 1),
            )


def _config_to_options(config: rasr.RecognitionConfig, settings: ServiceSettings) -> RecognitionOptions:
    language_code = config.language_code or settings.default_language_code
    if language_code.lower() != settings.default_language_code.lower():
        LOGGER.warning("Requested language code %s differs from default %s", language_code, settings.default_language_code)
    diarization = False
    max_speaker_count = max(1, settings.diarization_max_speakers)
    if config.HasField("diarization_config"):
        diarization = bool(config.diarization_config.enable_speaker_diarization)
        requested_max = getattr(config.diarization_config, "max_speaker_count", max_speaker_count)
        try:
            max_speaker_count = max(1, int(requested_max))
        except Exception:
            max_speaker_count = max(1, settings.diarization_max_speakers)
    options = RecognitionOptions(
        sample_rate=config.sample_rate_hertz or 0,
        language_code=language_code,
        include_word_time_offsets=bool(config.enable_word_time_offsets),
        enable_punctuation=bool(config.enable_automatic_punctuation or settings.default_punctuation),
        enable_diarization=bool(diarization or settings.default_diarization),
        interim_results=True,
        channel_count=max(1, config.audio_channel_count or 1),
        max_speaker_count=max_speaker_count,
        endpointing=_endpointing_from_config(config, settings),
    )
    return options


def _triton_model_metadata(config: TritonModelConfig) -> service_pb2.ModelMetadataResponse:
    response = service_pb2.ModelMetadataResponse()
    response.name = config.name
    response.platform = "riva-asr"
    response.versions.extend([config.version])
    tensor_input = response.inputs.add()
    tensor_input.name = "AUDIO"
    tensor_input.datatype = "BYTES"
    tensor_input.shape.extend([-1])
    tensor_output = response.outputs.add()
    tensor_output.name = "TRANSCRIPTS"
    tensor_output.datatype = "BYTES"
    tensor_output.shape.extend([-1])
    return response


def _triton_model_config_response(
    config: TritonModelConfig,
    settings: ServiceSettings,
) -> model_config_pb2.ModelConfigResponse:
    response = model_config_pb2.ModelConfigResponse()
    response.config.name = config.name
    response.config.platform = "riva-asr"
    response.config.max_batch_size = max(1, config.max_batch_size)
    model_input = response.config.input.add()
    model_input.name = "AUDIO"
    model_input.data_type = model_config_pb2.TYPE_BYTES
    model_input.dims.extend([-1])
    model_output = response.config.output.add()
    model_output.name = "TRANSCRIPTS"
    model_output.data_type = model_config_pb2.TYPE_BYTES
    model_output.dims.extend([-1])
    response.config.parameters["language_code"].string_value = config.language_code
    response.config.parameters["sample_rate"].string_value = str(config.sample_rate)
    response.config.parameters["version"].string_value = config.version
    response.config.parameters["supports_punctuation"].string_value = "true"
    response.config.parameters["supports_diarization"].string_value = "true"
    response.config.parameters["max_concurrent_requests"].string_value = str(settings.max_concurrent_requests)
    response.config.parameters["max_streaming_sessions"].string_value = str(settings.max_streaming_sessions)
    response.config.parameters["endpoint_start_history_ms"].string_value = str(settings.endpoint_start_history_ms)
    response.config.parameters["endpoint_start_threshold"].string_value = str(settings.endpoint_start_threshold)
    response.config.parameters["endpoint_stop_history_ms"].string_value = str(settings.endpoint_stop_history_ms)
    response.config.parameters["endpoint_stop_threshold"].string_value = str(settings.endpoint_stop_threshold)
    response.config.parameters["endpoint_stop_history_eou_ms"].string_value = str(settings.endpoint_eou_history_ms)
    response.config.parameters["endpoint_stop_threshold_eou"].string_value = str(settings.endpoint_eou_threshold)
    response.config.parameters["vad_frame_ms"].string_value = str(settings.vad_frame_ms)
    return response


def _triton_repository_index_response(config: TritonModelConfig, loaded: bool) -> service_pb2.RepositoryIndexResponse:
    response = service_pb2.RepositoryIndexResponse()
    model_index = response.models.add()
    model_index.name = config.name
    model_index.version = config.version
    model_index.state = "READY" if loaded else "UNAVAILABLE"
    model_index.reason = "" if loaded else "Model is unloaded"
    return response


def _triton_server_metadata_response(settings: ServiceSettings) -> service_pb2.ServerMetadataResponse:
    response = service_pb2.ServerMetadataResponse()
    response.name = settings.model_name
    response.version = settings.model_version
    response.extensions.extend(["classification", "sequence", "riva-asr"])
    return response


def _triton_model_ready_response(loaded: bool) -> service_pb2.ModelReadyResponse:
    return service_pb2.ModelReadyResponse(ready=loaded)


def _triton_server_ready_response(loaded: bool) -> service_pb2.ServerReadyResponse:
    return service_pb2.ServerReadyResponse(ready=loaded)


def _triton_server_live_response() -> service_pb2.ServerLiveResponse:
    return service_pb2.ServerLiveResponse(live=True)


def _triton_model_statistics_response(config: TritonModelConfig, loaded: bool) -> service_pb2.ModelStatisticsResponse:
    response = service_pb2.ModelStatisticsResponse()
    if loaded:
        stats = response.model_stats.add()
        stats.name = config.name
        stats.version = config.version
    return response


class ParakeetServicer(rasr_grpc.RivaSpeechRecognitionServicer):
    def __init__(
        self,
        asr_bundle: ASRBundle,
        settings: ServiceSettings,
        metrics: Metrics,
        limiter: RequestLimiter,
        tracer: Optional[trace.Tracer],
    ):
        self.asr_bundle = asr_bundle
        self.settings = settings
        self.logger = logging.getLogger("parakeet-grpc")
        self.metrics = metrics
        self.limiter = limiter
        self.tracer = tracer

    def _build_response(self, result: rasr.SpeechRecognitionResult) -> rasr.RecognizeResponse:
        response = rasr.RecognizeResponse()
        response.results.append(result)
        return response

    def Recognize(self, request: rasr.RecognizeRequest, context) -> rasr.RecognizeResponse:  # noqa: N802
        try:
            with self.metrics.track("Recognize"), self.limiter.limit(self.limiter.general):
                span = self.tracer.start_span("Recognize") if self.tracer else None
                try:
                    config = request.config if request.config.ListFields() else rasr.RecognitionConfig()
                    options = _config_to_options(config, self.settings)
                    audio, duration_seconds = self.asr_bundle.decode_bytes(request.audio, config)
                    trimmed_audio, offset_seconds = _trim_audio_with_vad(
                        audio,
                        self.asr_bundle.sample_rate,
                        self.asr_bundle.vad_detector,
                        options.endpointing,
                    )
                    result = self.asr_bundle.recognize_from_audio(
                        trimmed_audio,
                        include_word_offsets=options.include_word_time_offsets,
                        enable_diarization=options.enable_diarization,
                        enable_punctuation=options.enable_punctuation,
                        max_speaker_count=options.max_speaker_count,
                        offset_seconds=offset_seconds,
                        original_duration_seconds=duration_seconds,
                    )
                    self.metrics.audio_seconds.labels(method="Recognize").inc(result.audio_processed)
                    if span:
                        span.set_attribute("audio.duration_seconds", result.audio_processed)
                    return self._build_response(result)
                except ValueError as exc:
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.exception("recognition failed: %s", exc)
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INTERNAL, "recognition failed")
                finally:
                    if span and span.is_recording():
                        span.end()
        except BusyError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

    def StreamingRecognize(  # noqa: N802
        self,
        request_iterator: Iterable[rasr.StreamingRecognizeRequest],
        context,
    ) -> Iterator[rasr.StreamingRecognizeResponse]:
        try:
            with self.metrics.track("StreamingRecognize"), self.limiter.limit(self.limiter.streaming):
                span = self.tracer.start_span("StreamingRecognize") if self.tracer else None
                try:
                    config = rasr.RecognitionConfig()
                    interim_results = True
                    state: Optional[StreamingState] = None
                    config_seen = False
                    for message in request_iterator:
                        if message.HasField("streaming_config"):
                            config = message.streaming_config.config
                            interim_results = bool(message.streaming_config.interim_results)
                            state = StreamingState(options=_config_to_options(config, self.settings))
                            state.last_emit_time = time.perf_counter()
                            config_seen = True
                            continue
                        if not message.audio_content:
                            continue
                        if not config_seen and config.encoding == rasr.RecognitionConfig.AudioEncoding.LINEAR_PCM:
                            raise ValueError("streaming_config with sample_rate_hertz is required before PCM audio")
                        chunk_audio, duration_seconds = self.asr_bundle.decode_bytes(message.audio_content, config)
                        logits_chunk = self.asr_bundle._logits_from_audio(chunk_audio)
                        vad_probs, frame_duration = self.asr_bundle.vad_detector.probabilities(
                            chunk_audio,
                            self.asr_bundle.sample_rate,
                        )
                        if state is None:
                            state = StreamingState(options=_config_to_options(config, self.settings))
                            state.last_emit_time = time.perf_counter()
                        state.append(chunk_audio, logits_chunk, duration_seconds)
                        _update_endpointing_state(
                            state,
                            state.options.endpointing,
                            vad_probs,
                            frame_duration,
                        )
                        self.metrics.audio_seconds.labels(method="StreamingRecognize").inc(duration_seconds)
                        now = time.perf_counter()
                        pipeline_vad = _pipeline_vad_probabilities(state)
                        emit_interval_reached = (now - state.last_emit_time) * 1000.0 >= float(self.settings.streaming_emit_interval_ms)
                        if interim_results and state.speech_started and emit_interval_reached and not state.endpoint_triggered:
                            partial = self._build_streaming_response(
                                state,
                                is_final=False,
                                pipeline_vad=pipeline_vad,
                            )
                            if partial and partial.results and partial.results[0].alternatives:
                                transcript = partial.results[0].alternatives[0].transcript
                                if transcript != state.last_transcript:
                                    yield partial
                                    state.last_transcript = transcript
                                state.last_emit_time = now
                        if state.endpoint_triggered and not state.final_emitted:
                            final_response = self._build_streaming_response(
                                state,
                                is_final=True,
                                pipeline_vad=pipeline_vad,
                            )
                            if span:
                                span.set_attribute("audio.duration_seconds", state.total_audio_seconds)
                            if final_response:
                                yield final_response
                            state.final_emitted = True
                            break
                    if state:
                        if span:
                            span.set_attribute("audio.duration_seconds", state.total_audio_seconds)
                        if state.final_emitted:
                            return
                        final_response = self._build_streaming_response(
                            state,
                            is_final=True,
                            pipeline_vad=_pipeline_vad_probabilities(state),
                        )
                        if final_response:
                            yield final_response
                except ValueError as exc:
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.exception("streaming recognition failed: %s", exc)
                    if span:
                        span.record_exception(exc)
                        span.end()
                    context.abort(grpc.StatusCode.INTERNAL, "streaming recognition failed")
                finally:
                    if span and span.is_recording():
                        span.end()
        except BusyError as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))

    def _build_streaming_response(
        self,
        state: StreamingState,
        is_final: bool,
        pipeline_vad: Optional[Sequence[float]] = None,
    ) -> Optional[rasr.StreamingRecognizeResponse]:
        merged_logits = state.merged_logits()
        if merged_logits is None:
            return None
        transcript, offsets = self.asr_bundle._decode_logits(
            merged_logits,
            state.options.include_word_time_offsets,
        )
        if state.options.enable_punctuation:
            transcript = _apply_punctuation(transcript, self.asr_bundle._ensure_punctuation_restorer())
        speaker_assignments: Optional[Dict[int, int]] = None
        if state.options.enable_diarization and state.options.include_word_time_offsets:
            merged_audio = state.merged_audio()
            if merged_audio is not None:
                segments = _estimate_speaker_segments(
                    merged_audio,
                    self.asr_bundle.sample_rate,
                    max(1, state.options.max_speaker_count),
                )
                speaker_assignments = _assign_speakers(
                    offsets,
                    segments,
                    self.asr_bundle.time_offset_seconds,
                )
        streaming_response = rasr.StreamingRecognizeResponse()
        response_result = streaming_response.results.add()
        alternative = rasr.SpeechRecognitionAlternative()
        alternative.transcript = transcript
        alternative.confidence = 0.0
        if state.options.include_word_time_offsets:
            alternative.words.extend(
                self.asr_bundle._word_infos(
                    offsets,
                    speaker_assignments,
                )
            )
        response_result.alternatives.append(alternative)
        response_result.is_final = is_final
        response_result.channel_tag = 0
        response_result.audio_processed = state.total_audio_seconds
        if not is_final:
            response_result.stability = 0.9
        else:
            response_result.stability = 1.0
        if pipeline_vad:
            pipeline_states = rasr.PipelineStates()
            pipeline_states.vad_probabilities.extend(pipeline_vad)
            response_result.pipeline_states.CopyFrom(pipeline_states)
        return streaming_response

    def GetRivaSpeechRecognitionConfig(  # noqa: N802
        self,
        request: rasr.RivaSpeechRecognitionConfigRequest,
        context,
    ) -> rasr.RivaSpeechRecognitionConfigResponse:
        response = rasr.RivaSpeechRecognitionConfigResponse()
        config = response.model_config.add()
        config.model_name = self.settings.model_name
        config.parameters["version"] = self.settings.model_version
        config.parameters["sample_rate_hz"] = str(self.asr_bundle.sample_rate)
        config.parameters["language_code"] = self.settings.default_language_code
        config.parameters["max_batch_size"] = str(self.settings.max_batch_size)
        config.parameters["max_concurrent_requests"] = str(self.settings.max_concurrent_requests)
        config.parameters["max_streaming_sessions"] = str(self.settings.max_streaming_sessions)
        config.parameters["supports_punctuation"] = "true"
        config.parameters["supports_diarization"] = "true"
        config.parameters["default_punctuation"] = str(self.settings.default_punctuation)
        config.parameters["default_diarization"] = str(self.settings.default_diarization)
        config.parameters["endpoint_start_history_ms"] = str(self.settings.endpoint_start_history_ms)
        config.parameters["endpoint_start_threshold"] = str(self.settings.endpoint_start_threshold)
        config.parameters["endpoint_stop_history_ms"] = str(self.settings.endpoint_stop_history_ms)
        config.parameters["endpoint_stop_threshold"] = str(self.settings.endpoint_stop_threshold)
        config.parameters["endpoint_stop_history_eou_ms"] = str(self.settings.endpoint_eou_history_ms)
        config.parameters["endpoint_stop_threshold_eou"] = str(self.settings.endpoint_eou_threshold)
        config.parameters["vad_frame_ms"] = str(self.settings.vad_frame_ms)
        return response


class TritonControlService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, config: TritonModelConfig, settings: ServiceSettings):
        self.config = config
        self.settings = settings
        self.loaded = True

    def _validate_model(self, name: Optional[str], context) -> None:
        if name and name not in {self.config.name}:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Model {name} not found")

    def ServerLive(self, request, context):  # noqa: N802
        return _triton_server_live_response()

    def ServerReady(self, request, context):  # noqa: N802
        return _triton_server_ready_response(self.loaded)

    def ServerMetadata(self, request, context):  # noqa: N802
        return _triton_server_metadata_response(self.settings)

    def ModelReady(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return _triton_model_ready_response(self.loaded)

    def ModelMetadata(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return _triton_model_metadata(self.config)

    def ModelConfig(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return _triton_model_config_response(self.config, self.settings)

    def RepositoryIndex(self, request, context):  # noqa: N802
        return _triton_repository_index_response(self.config, self.loaded)

    def RepositoryModelLoad(self, request, context):  # noqa: N802
        self.loaded = True
        return service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(self, request, context):  # noqa: N802
        self.loaded = False
        return service_pb2.RepositoryModelUnloadResponse()

    def ModelStatistics(self, request, context):  # noqa: N802
        self._validate_model(request.name, context)
        return _triton_model_statistics_response(self.config, self.loaded)


class AuthInterceptor(grpc.ServerInterceptor):
    def __init__(self, validator: AuthValidator):
        self.validator = validator

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or not self.validator.enabled:
            return handler

        def _metadata(context) -> Dict[str, str]:
            return {key.lower(): value for key, value in (context.invocation_metadata() or [])}

        def _enforce_auth(context) -> None:
            try:
                self.validator.validate(_metadata(context))
            except PermissionError as exc:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))

        if handler.unary_unary:
            def unary_unary(request, context):
                _enforce_auth(context)
                return handler.unary_unary(request, context)
            return grpc.unary_unary_rpc_method_handler(
                unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_unary:
            def stream_unary(request_iter, context):
                _enforce_auth(context)
                return handler.stream_unary(request_iter, context)
            return grpc.stream_unary_rpc_method_handler(
                stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.unary_stream:
            def unary_stream(request, context):
                _enforce_auth(context)
                return handler.unary_stream(request, context)
            return grpc.unary_stream_rpc_method_handler(
                unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_stream:
            def stream_stream(request_iter, context):
                _enforce_auth(context)
                return handler.stream_stream(request_iter, context)
            return grpc.stream_stream_rpc_method_handler(
                stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler


def _start_http(
    settings: ServiceSettings,
    metrics: Metrics,
    auth_validator: AuthValidator,
    start_time: float,
    triton_config: Optional[TritonModelConfig],
    triton_service: Optional["TritonControlService"],
) -> Optional[ThreadingHTTPServer]:
    if settings.http_port <= 0:
        return None

    class _Handler(BaseHTTPRequestHandler):
        server_version = "parakeet-open"

        def _metadata(self) -> Dict[str, str]:
            return {key.lower(): value for key, value in self.headers.items()}

        def _ensure_auth(self) -> bool:
            try:
                auth_validator.validate(self._metadata(), allow_unauthenticated=settings.allow_unauth_health and self.path in {"/", "/v1/health/live", "/v1/health/ready"})
            except PermissionError:
                self.send_error(HTTPStatus.UNAUTHORIZED, "authentication required")
                return False
            return True

        def _send_json(self, payload: Dict[str, object], status_code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if not self._ensure_auth():
                return
            if self.path in {"/", "/v1/health/live"}:
                self._send_json({"status": "SERVING"})
            elif self.path == "/v1/health/ready":
                self._send_json(
                    {
                        "status": "READY",
                        "uptime_seconds": int(time.time() - start_time),
                        "model": settings.model_name,
                        "version": settings.model_version,
                    }
                )
            elif self.path == "/v1/metadata":
                short_name = f"{settings.model_name}:{settings.model_version}"
                self._send_json(
                    {
                        "id": settings.model_name,
                        "name": settings.model_name,
                        "version": settings.model_version,
                        "limits": {
                            "maxBatchSize": settings.max_batch_size,
                            "maxConcurrentRequests": settings.max_concurrent_requests,
                            "maxStreamingSessions": settings.max_streaming_sessions,
                        },
                        "features": {
                            "languageCode": settings.default_language_code,
                            "punctuation": True,
                            "punctuationDefault": settings.default_punctuation,
                            "diarization": True,
                            "diarizationDefault": settings.default_diarization,
                            "endpointing": {
                                "startHistoryMs": settings.endpoint_start_history_ms,
                                "startThreshold": settings.endpoint_start_threshold,
                                "stopHistoryMs": settings.endpoint_stop_history_ms,
                                "stopThreshold": settings.endpoint_stop_threshold,
                                "eouHistoryMs": settings.endpoint_eou_history_ms,
                                "eouThreshold": settings.endpoint_eou_threshold,
                                "vadFrameMs": settings.vad_frame_ms,
                            },
                        },
                        "modelInfo": [
                            {
                                "name": settings.model_name,
                                "version": settings.model_version,
                                "shortName": short_name,
                            }
                        ],
                    }
                )
            elif self.path == "/metrics":
                body = generate_latest(metrics.registry)
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", CONTENT_TYPE_LATEST)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in {"/v2/health/live"}:
                payload = MessageToDict(
                    _triton_server_live_response(),
                    preserving_proto_field_name=True,
                )
                self._send_json(payload)
            elif self.path in {"/v2/health/ready"}:
                payload = MessageToDict(
                    _triton_server_ready_response(triton_service.loaded if triton_service else True),
                    preserving_proto_field_name=True,
                )
                self._send_json(payload)
            elif self.path.startswith("/v2/repository/index"):
                if not triton_config:
                    self._send_json({"error": "not found"}, status_code=404)
                    return
                payload = MessageToDict(
                    _triton_repository_index_response(triton_config, triton_service.loaded if triton_service else True),
                    preserving_proto_field_name=True,
                )
                self._send_json(payload)
            elif self.path.startswith("/v2/models/"):
                if not triton_config:
                    self._send_json({"error": "not found"}, status_code=404)
                    return
                parts = [part for part in self.path.split("/") if part]
                if len(parts) < 3:
                    self._send_json({"error": "not found"}, status_code=404)
                    return
                model_name = parts[2]
                if model_name != triton_config.name:
                    self._send_json({"error": "model not found"}, status_code=404)
                    return
                if len(parts) >= 4 and parts[3] == "config":
                    payload = MessageToDict(
                        _triton_model_config_response(triton_config, settings),
                        preserving_proto_field_name=True,
                    )
                    self._send_json(payload)
                elif len(parts) >= 4 and parts[3] == "ready":
                    payload = MessageToDict(
                        _triton_model_ready_response(triton_service.loaded if triton_service else True),
                        preserving_proto_field_name=True,
                    )
                    self._send_json(payload)
                else:
                    payload = MessageToDict(
                        _triton_model_metadata(triton_config),
                        preserving_proto_field_name=True,
                    )
                    self._send_json(payload)
            else:
                self._send_json({"error": "not found"}, status_code=404)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("0.0.0.0", settings.http_port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def serve() -> None:
    settings = ServiceSettings()
    logging.basicConfig(level=settings.log_level.upper())
    tracer = _setup_tracer(settings)
    metrics = Metrics(namespace=settings.metrics_namespace)
    asr_bundle = ASRBundle(settings)
    triton_config = TritonModelConfig(
        name=settings.model_name,
        version=settings.model_version,
        max_batch_size=settings.max_batch_size,
        sample_rate=asr_bundle.sample_rate,
        language_code=settings.default_language_code,
    )
    limiter = RequestLimiter(settings.max_concurrent_requests, settings.max_streaming_sessions)
    auth_validator = AuthValidator(settings)

    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.max_workers),
        interceptors=[AuthInterceptor(auth_validator)],
        options=[
            ("grpc.max_concurrent_streams", settings.max_streaming_sessions),
        ],
    )
    rasr_grpc.add_RivaSpeechRecognitionServicer_to_server(
        ParakeetServicer(asr_bundle, settings, metrics, limiter, tracer),
        grpc_server,
    )
    triton_service = TritonControlService(triton_config, settings)
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(triton_service, grpc_server)
    grpc_server.add_insecure_port(f"[::]:{settings.grpc_port}")

    start_time = time.time()
    http_server = _start_http(settings, metrics, auth_validator, start_time, triton_config, triton_service)
    if http_server:
        LOGGER.info("HTTP health/metadata listening on %s", settings.http_port)

    grpc_server.start()
    LOGGER.info("gRPC listening on %s", settings.grpc_port)

    def _graceful_shutdown(*_: object) -> None:
        grpc_server.stop(grace=None)
        if http_server:
            http_server.shutdown()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    serve()
