# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (bytes, bytearray)):
        return {"omitted_bytes": int(len(value))}

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key == "image_b64" and isinstance(raw_value, str):
                out["image_b64_chars"] = int(len(raw_value))
                out["image_b64_omitted"] = True
                continue
            if key == "embedding":
                try:
                    dim = len(raw_value)  # type: ignore[arg-type]
                except Exception:
                    dim = None
                out["embedding_dimensions"] = int(dim) if dim is not None else None
                out["embedding_omitted"] = True
                continue
            out[key] = _to_jsonable(raw_value)
        return out

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    return str(value)


def _normalize_record(record: dict[str, Any], *, drop_columns: set[str]) -> dict[str, Any]:
    return {str(key): _to_jsonable(value) for key, value in record.items() if str(key) not in drop_columns}


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    tmp_path.replace(path)


def write_jsonl_snapshot_batch(
    batch_df: Any,
    *,
    output_dir: str,
    stage_name: str,
    durable_output_dir: str | None = None,
    drop_columns: list[str] | None = None,
) -> Any:
    raw_records = batch_df.to_dict(orient="records")
    if not raw_records:
        return batch_df

    records = [
        _normalize_record(record, drop_columns={str(col) for col in (drop_columns or [])}) for record in raw_records
    ]
    if not records:
        return batch_df

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    file_name = f"{stage_name}-{timestamp}-{uuid4().hex[:8]}.jsonl"
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)

    primary_path = Path(output_dir).expanduser().resolve() / file_name
    _atomic_write_text(primary_path, payload)

    if durable_output_dir:
        durable_path = Path(durable_output_dir).expanduser().resolve() / file_name
        if durable_path != primary_path:
            try:
                _atomic_write_text(durable_path, payload)
            except Exception:
                logger.warning("Failed writing durable observability snapshot to %s", durable_path, exc_info=True)

    return batch_df
