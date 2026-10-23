# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, bounded, private HF embedding fingerprints, separate from model identity.

The context is per call, never installed on a warmed/shared model. YAML contains
hashes and structural metadata only. Hashes remain sensitive/linkable evidence.
Capturing inputs adds CPU work and may synchronize CUDA; use diagnostic runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable
import uuid

import yaml

from nemo_retriever.common.params import EmbedInputAuditParams

_ACTIVE: ContextVar[tuple | None] = ContextVar("embed_input_audit", default=None)
_ROWS: ContextVar[AuditRows | None] = ContextVar("embed_input_audit_rows", default=None)


def _fresh_values(value: Any) -> Any:
    # A canonical value must not depend on whether Python containers are shared.
    # Fresh containers prevent SafeDumper aliases without accepting new types.
    if isinstance(value, dict):
        return {key: _fresh_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fresh_values(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    """nrl-safe-yaml-v1: sorted safe YAML, explicit documents, UTF-8 and LF.

    Exact flags and the PyYAML version are recorded in each shard header. This
    encoding applies only to private fingerprint metadata.
    """
    return yaml.safe_dump(
        _fresh_values(value),
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        explicit_start=True,
        explicit_end=True,
        indent=2,
        width=4096,
        line_break="\n",
        encoding="utf-8",
    )


def _identity_value(value: Any) -> Any:
    """Normalize only the small selected identity fields, without repr output."""
    if isinstance(value, dict):
        return {key: _identity_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _identity_value(value.item())
        except (ValueError, TypeError):
            return None
    return None


def _digest(domain: str, data: bytes) -> str:
    h = hashlib.sha256()
    for part in (b"nrl-input-audit-v1", domain.encode(), data):
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


class _BudgetExceeded(Exception):
    pass


class InputAuditSink:
    """One actor owns one private multi-document YAML shard and finite budgets."""

    def __init__(self, params: EmbedInputAuditParams):
        self.params = params
        self.active = True
        self.records = 0
        self.hashed_bytes = 0
        self.output_bytes = 0
        self.sequence = 0
        self._headers: set[int] = set()
        self._lock = threading.RLock()
        destination = Path(params.output_dir)
        destination.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = destination.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("embed_input_audit output_dir must be an owned private directory (mode 0700).")
        self.path = destination / f"hf-input-{os.getpid()}-{uuid.uuid4().hex}.yaml"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags, 0o600)
        os.close(fd)
        self._emit(
            {
                "event": "header",
                "schema_version": 1,
                "complete": False,
                "canonicalization": {
                    "scheme": "nrl-safe-yaml-v1",
                    "sort_keys": True,
                    "allow_unicode": True,
                    "default_flow_style": False,
                    "explicit_start": True,
                    "explicit_end": True,
                    "indent": 2,
                    "width": 4096,
                    "line_break": "LF",
                    "encoding": "utf-8",
                    "aliases": False,
                },
                "yaml_serializer_version": yaml.__version__,
                "budgets_per_actor": params.model_dump(exclude={"output_dir"}),
                "timing_qualification": False,
            }
        )

    @staticmethod
    def validate_model_config(config: dict) -> None:
        """Reject unsupported diagnostic paths before any model is loaded."""
        if (
            config.get("embedding_endpoint")
            or config.get("embed_invoke_url")
            or config.get("local_ingest_embed_backend", "vllm") != "hf"
        ):
            raise ValueError("embed_input_audit supports only the local HF Nemotron 3.5 adapter.")
        from nemo_retriever.models.local_embedder_spec import LocalEmbedderSpec

        spec = LocalEmbedderSpec.from_config(config)
        if spec.backend != "hf" or spec.checkpoint.model_type != "mistral3":
            raise ValueError("embed_input_audit supports only the local HF Nemotron 3.5 adapter.")

    def _emit(self, value: dict, *, terminal: bool = False) -> None:
        payload = yaml.safe_dump({"sequence": self.sequence, **value}, sort_keys=False, explicit_start=True).encode()
        reserve = 1024 if not terminal else 0
        if self.output_bytes + len(payload) + reserve > self.params.max_output_bytes:
            raise _BudgetExceeded("output_bytes")
        with self.path.open("ab") as stream:
            stream.write(payload)
        self.sequence += 1
        self.output_bytes += len(payload)

    def _stop(self, reason: str, error_type: str | None = None) -> None:
        self.active = False
        try:
            self._emit(
                {
                    "event": "audit_stopped",
                    "complete": False,
                    "reason": reason,
                    "error_type": error_type,
                    "admitted_rows": self.records,
                    "hashed_bytes": self.hashed_bytes,
                },
                terminal=True,
            )
        except Exception:
            # A missing final receipt must be interpreted as incomplete too.
            pass

    def capture(self, event: str, build: Callable[[], dict]) -> None:
        with self._lock:
            if not self.active:
                return
            try:
                state = _ACTIVE.get()
                self._emit({"event": event, "call_id": state[2] if state else None, **build()})
            except _BudgetExceeded as exc:
                self._stop(str(exc))
            except Exception as exc:
                # Exception messages can contain source text, URLs or paths.
                self._stop("capture_error", type(exc).__name__)

    def bytes_hash(self, domain: str, value: bytes) -> str:
        self._charge(len(value))
        return _digest(domain, value)

    def _charge(self, count: int) -> None:
        if self.hashed_bytes + count > self.params.max_hashed_bytes:
            raise _BudgetExceeded("hashed_bytes")
        self.hashed_bytes += count

    def tensor(self, value) -> dict:
        """Hash logical C-order exact bits in bounded slabs, preserving BF16."""
        import torch

        info = {"dtype": str(value.dtype), "shape": list(value.shape)}
        byte_count = value.numel() * value.element_size()
        self._charge(byte_count)
        h = hashlib.sha256()
        header = _canonical({"domain": "nrl-input-audit-tensor-v1", "byte_order": "little", **info})
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        # CUDA deployments and supported CPU runners are little-endian. Reject
        # other hosts rather than silently changing the canonical contract.
        import sys

        if sys.byteorder != "little":
            raise ValueError("Unsupported tensor byte order")

        def visit(tensor):
            if tensor.is_contiguous():
                flat = tensor.detach().reshape(-1)
                step = max(1, self.params.chunk_bytes // tensor.element_size())
                for start in range(0, flat.numel(), step):
                    slab = flat[start : start + step].cpu().contiguous().view(torch.uint8)
                    h.update(memoryview(slab.numpy()))
            else:
                for child in tensor.unbind(0):
                    visit(child)

        visit(value)
        return {**info, "sha256": h.hexdigest(), "byte_count": byte_count}

    @contextmanager
    def activate(self, *, role: str):
        call_id = uuid.uuid4().hex
        role = "query" if str(role).strip().lower() == "query" else "passage"
        token = _ACTIVE.set((self, role, call_id))
        self.capture("call_start", lambda: {"role": role})
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            self.capture(
                "call_end",
                lambda: {"complete": succeeded, "admitted_rows": self.records, "hashed_bytes": self.hashed_bytes},
            )
            _ACTIVE.reset(token)

    def model_header(self, adapter) -> None:
        key = id(adapter)
        if key in self._headers:
            return

        def build():
            import torch

            config = adapter._model.config
            self._headers.add(key)
            from importlib.metadata import PackageNotFoundError, version

            parameters = {(str(p.device), str(p.dtype)) for p in adapter._model.parameters()}
            versions = {}
            for package in ("transformers", "tokenizers", "Pillow", "numpy"):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    versions[package] = None
            tokenizer = adapter._processor.tokenizer

            return {
                "model_id": adapter.model_id,
                "declared_revision": adapter.revision,
                "resolved_config_commit": getattr(config, "_commit_hash", None),
                "weight_hash_verified_here": False,
                "model_config_sha256": self.bytes_hash("model-config", _canonical(config.to_dict())),
                "template_sha256": self.bytes_hash("template", adapter._template.encode()),
                "processor_class": type(adapter._processor).__qualname__,
                "processor_config_sha256": self.bytes_hash(
                    "processor-config", _canonical(adapter._processor.to_dict())
                ),
                "torch_version": str(torch.__version__),
                "library_versions": versions,
                "tokenizer_class": type(tokenizer).__qualname__,
                "tokenizer_special_tokens_sha256": self.bytes_hash(
                    "special-tokens", _canonical(getattr(tokenizer, "special_tokens_map", {}))
                ),
                "model_device": str(adapter._model.device),
                "parameter_device_dtypes": [list(pair) for pair in sorted(parameters)],
                "training": bool(adapter._model.training),
                "attention_implementation": getattr(config, "_attn_implementation", None),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            }

        self.capture("model_identity", build)


@dataclass
class AuditRows:
    sink: InputAuditSink
    identities: list[str]
    image_ordinal: int = 0
    output_rows: int = 0
    processor_fields: dict | None = None

    def image(self, encoded: bytes, image) -> None:
        ordinal = self.image_ordinal
        self.image_ordinal += 1

        def build():
            h = hashlib.sha256()
            header = _canonical({"domain": "RGB-v1", "mode": image.mode, "size": list(image.size)})
            h.update(len(header).to_bytes(8, "little"))
            h.update(header)
            width, height = image.size
            row_bytes = width * len(image.getbands())
            self.sink._charge(row_bytes * height)
            step = max(1, self.sink.params.chunk_bytes // max(1, row_bytes))
            if row_bytes > self.sink.params.chunk_bytes:
                raise _BudgetExceeded("image_row_bytes")
            for top in range(0, height, step):
                with image.crop((0, top, width, min(height, top + step))) as strip:
                    h.update(strip.tobytes())
            return {
                "identity": self.identities[ordinal],
                "image_ordinal": ordinal,
                "compressed_sha256": self.sink.bytes_hash("compressed-image", encoded),
                "compressed_byte_count": len(encoded),
                "size": [width, height],
                "mode": image.mode,
                "rgb_sha256": h.hexdigest(),
            }

        self.sink.capture("decoded_image", build)

    def processor(self, batch, texts, prompts, start: int) -> None:
        def build():
            import torch

            fields = {}
            for name, value in sorted(batch.items()):
                fields[name] = (
                    self.sink.tensor(value)
                    if isinstance(value, torch.Tensor)
                    else {
                        "sha256": self.sink.bytes_hash("processor-value", _canonical(value)),
                        "type": type(value).__name__,
                    }
                )
            self.processor_fields = fields
            valid = []
            for position in range(len(prompts)):
                item = {
                    "identity": self.identities[start + position],
                    "text_sha256": self.sink.bytes_hash("effective-text", str(texts[position]).encode()),
                    "prompt_sha256": self.sink.bytes_hash("prompt", prompts[position].encode()),
                    "token_ids": self.sink.tensor(
                        batch["input_ids"][position][batch["attention_mask"][position].bool()]
                    ),
                }
                if "pixel_values" in batch:
                    size = batch["image_sizes"][position]
                    height, width = int(size[0]), int(size[1])
                    item["valid_pixels"] = self.sink.tensor(batch["pixel_values"][position, :, :height, :width])
                valid.append(item)
            return {
                "ordered_identities": self.identities[start : start + len(prompts)],
                "padded_processor_fields": fields,
                "valid_items": valid,
            }

        self.sink.capture("processor", build)

    def model_call(self, batch, start: int, count: int) -> None:
        def build():
            import torch

            fields = dict(self.processor_fields or {})
            for name, value in batch.items():
                if name not in fields:
                    fields[name] = (
                        self.sink.tensor(value)
                        if isinstance(value, torch.Tensor)
                        else {"sha256": self.sink.bytes_hash("model-value", _canonical(value))}
                    )
                elif isinstance(value, torch.Tensor) and (
                    list(value.shape) != fields[name].get("shape") or str(value.dtype) != fields[name].get("dtype")
                ):
                    raise ValueError("Processor transfer changed shape or dtype")
            if set(fields) != set(batch):
                raise ValueError("Model kwargs differ from processor snapshot")
            ordered = self.identities[start : start + count]
            return {
                "ordered_identities": ordered,
                "fields": fields,
                "device_by_field": {k: str(v.device) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                "input_sha256": _digest("model-call", _canonical({"members": ordered, "fields": fields})),
                "tensor_byte_observation": "CPU processor buffers before device-only transfer; actual derived kwargs",
            }

        self.sink.capture("model_call", build)

    def outputs(self, pooled, final, start: int) -> None:
        def build():
            items = [
                {
                    "identity": self.identities[start + i],
                    "pooled": self.sink.tensor(pooled[i]),
                    "final": self.sink.tensor(final[i]),
                    "norm": float(final[i].norm()),
                    "finite": bool(final[i].isfinite().all()),
                }
                for i in range(len(final))
            ]
            self.output_rows += len(items)
            return {"items": items}

        self.sink.capture("model_output", build)


def current_rows() -> AuditRows | None:
    value = _ROWS.get()
    return value if value is not None and value.sink.active else None


@contextmanager
def audit_rows(frame, *, text_column: str, admit: bool = True):
    """Bind exact DataFrame row identities to one real model invocation."""
    state = _ACTIVE.get()
    if state is None or not state[0].active:
        yield
        return
    sink, role, _ = state
    identities = []

    def build():
        from nemo_retriever.common.schemas.embedding import select_embedding_text

        records = []
        for ordinal, (_, row) in enumerate(frame.iterrows()):
            if admit:
                if sink.records >= sink.params.max_records:
                    raise _BudgetExceeded("records")
                sink.records += 1
            metadata = row.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            source = metadata.get("source_metadata") or {}
            content = metadata.get("content_metadata") or {}
            split = metadata.get("embedding_split") or {}
            selected = select_embedding_text(row, text_column=text_column)
            text = selected.content if selected else ""
            text_hash = sink.bytes_hash("selected-text", text.encode())
            components = {
                "source": {k: source.get(k) for k in ("source_id", "source_name")},
                "content": {
                    k: content.get(k)
                    for k in ("page_number", "type", "subtype", "element_type", "bbox", "bounding_box")
                },
                "row": {k: row.get(k) for k in ("source_id", "source_name", "page_number", "element_id")},
                "split": {k: split.get(k) for k in ("parent_id", "chunk_id", "chunk_index")},
            }
            # Missing/NaN scalar identity values are normalized explicitly.
            components = _identity_value(components)
            has_source_identity = any(v is not None for group in components.values() for v in group.values())
            if not has_source_identity:
                components = {"fallback_text_sha256": text_hash}

            identity_basis = (
                "source_page_element" if has_source_identity else "text_fallback_not_stable_under_text_change"
            )
            query_index = row.get("_nrl_query_index")
            if role == "query":
                identity_basis = "query_position_and_text_hash_external_query_id_not_available"
                components = {
                    "query_index": int(query_index) if query_index is not None else None,
                    "query_text_sha256": text_hash,
                }
            identity = sink.bytes_hash("row-identity", _canonical(components))
            identities.append(identity)
            records.append(
                {
                    "identity": identity,
                    "identity_basis": identity_basis,
                    "ordinal": ordinal,
                    "selected_text_sha256": text_hash,
                    "selected_text_bytes": len(text.encode()),
                    "selected_column": selected.column if selected else None,
                    "has_image": isinstance(row.get("_image_b64"), str) and bool(row.get("_image_b64")),
                }
            )
        return {"role": role, "items": records}

    sink.capture("row_inputs", build)
    context = AuditRows(sink, identities) if sink.active else None
    token = _ROWS.set(context)
    try:
        yield
    finally:
        if admit and context is not None and sink.active and context.output_rows != len(identities):
            with sink._lock:
                sink._stop("missing_model_outputs")
        _ROWS.reset(token)


def invoke_with_rows(frame, invoke, *, text_column: str, positions=None):
    """Avoid DataFrame selection and all hashing when diagnostics are off."""
    state = _ACTIVE.get()
    if state is None or not state[0].active:
        return invoke()
    frame = frame() if callable(frame) else frame
    positions = positions() if callable(positions) else positions
    selected = frame if positions is None else frame.iloc[positions]
    with audit_rows(selected, text_column=text_column):
        return invoke()


def audit_handoff(frame, *, output_column: str, text_column: str) -> None:
    """Fingerprint row-to-vector attachment before the VDB writer sees it."""
    state = _ACTIVE.get()
    if state is None or not state[0].active:
        return
    sink = state[0]
    with audit_rows(frame, text_column=text_column, admit=False):
        rows = current_rows()
        if rows is None:
            return

        def build():
            import torch

            items = []
            for identity, (_, row) in zip(rows.identities, frame.iterrows()):
                payload = row.get(output_column)
                vector = payload.get("embedding") if isinstance(payload, dict) else None
                if vector is None:
                    metadata = row.get("metadata")
                    vector = metadata.get("embedding") if isinstance(metadata, dict) else None
                items.append(
                    {
                        "identity": identity,
                        "vector": (
                            sink.tensor(torch.as_tensor(vector, dtype=torch.float32)) if vector is not None else None
                        ),
                    }
                )
            return {"items": items}

        sink.capture("row_handoff", build)
