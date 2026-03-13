# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NEMO_RETRIEVER_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = NEMO_RETRIEVER_ROOT.parent
DEFAULT_TEST_CONFIG_PATH = NEMO_RETRIEVER_ROOT / "harness" / "test_configs.yaml"
DEFAULT_NIGHTLY_CONFIG_PATH = NEMO_RETRIEVER_ROOT / "harness" / "nightly_config.yaml"
VALID_RECALL_ADAPTERS = {"none", "page_plus_one", "financebench_json"}
AUTO_WORKER_FIELDS = {
    "pdf_extract_workers",
    "page_elements_workers",
    "ocr_workers",
    "embed_workers",
}
DEFAULT_NIGHTLY_SLACK_METRIC_KEYS = [
    "pages",
    "ingest_secs",
    "pages_per_sec_ingest",
    "recall_5",
]

TUNING_FIELDS = {
    "pdf_extract_workers",
    "pdf_extract_num_cpus",
    "pdf_extract_batch_size",
    "pdf_split_batch_size",
    "page_elements_batch_size",
    "page_elements_workers",
    "ocr_workers",
    "ocr_batch_size",
    "embed_workers",
    "embed_batch_size",
    "page_elements_cpus_per_actor",
    "ocr_cpus_per_actor",
    "embed_cpus_per_actor",
    "gpu_page_elements",
    "gpu_ocr",
    "gpu_embed",
}


@dataclass
class HarnessConfig:
    dataset_dir: str
    dataset_label: str
    preset: str

    query_csv: str | None = None
    input_type: str = "pdf"
    recall_required: bool = True
    recall_match_mode: str = "pdf_page"
    recall_adapter: str = "none"

    artifacts_dir: str | None = None
    ray_address: str | None = None
    lancedb_uri: str = "lancedb"
    hybrid: bool = False
    embed_model_name: str = "nvidia/llama-nemotron-embed-1b-v2"
    write_detection_file: bool = False

    pdf_extract_workers: int | None = 8
    pdf_extract_num_cpus: float = 2.0
    pdf_extract_batch_size: int = 4
    pdf_split_batch_size: int = 1
    page_elements_batch_size: int = 4
    page_elements_workers: int | None = 3
    ocr_workers: int | None = 3
    ocr_batch_size: int = 16
    embed_workers: int | None = 3
    embed_batch_size: int = 256
    page_elements_cpus_per_actor: float = 1.0
    ocr_cpus_per_actor: float = 1.0
    embed_cpus_per_actor: float = 1.0
    gpu_page_elements: float = 0.1
    gpu_ocr: float = 0.1
    gpu_embed: float = 0.25

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dataset_dir:
            errors.append("dataset_dir is required")
        elif not Path(self.dataset_dir).exists():
            errors.append(f"dataset_dir does not exist: {self.dataset_dir}")

        if self.query_csv is not None and not Path(self.query_csv).exists():
            errors.append(f"query_csv does not exist: {self.query_csv}")

        if self.recall_required and not self.query_csv:
            errors.append("recall_required=true requires query_csv")

        if self.input_type not in {"pdf", "txt", "html", "doc"}:
            errors.append(f"input_type must be one of pdf/txt/html/doc, got '{self.input_type}'")

        if self.recall_match_mode not in {"pdf_page", "pdf_only"}:
            errors.append("recall_match_mode must be one of pdf_page/pdf_only")

        if self.recall_adapter not in VALID_RECALL_ADAPTERS:
            errors.append(f"recall_adapter must be one of {sorted(VALID_RECALL_ADAPTERS)}")

        for name in TUNING_FIELDS:
            val = getattr(self, name)
            if val is None and name in AUTO_WORKER_FIELDS:
                continue
            if name.startswith("gpu_") and float(val) < 0.0:
                errors.append(f"{name} must be >= 0.0")
            elif name.endswith("_workers") and int(val) < 1:
                errors.append(f"{name} must be >= 1")
            elif name.endswith("_batch_size") and int(val) < 1:
                errors.append(f"{name} must be >= 1")
            elif name.endswith("_num_cpus") and float(val) <= 0.0:
                errors.append(f"{name} must be > 0.0")

        return errors


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_number(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def _parse_auto_worker_value(value: str) -> int | None:
    if str(value).strip().lower() == "auto":
        return None
    return int(value)


def _resolve_config_path(config_file: str | None, default_path: Path) -> Path:
    if config_file is None:
        return default_path
    p = Path(config_file).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level: {path}")
    return data


def _resolve_preset_values(
    presets: dict[str, Any], preset_name: str, *, resolution_stack: tuple[str, ...] = ()
) -> dict[str, Any]:
    preset_key = str(preset_name)
    if preset_key not in presets:
        return {}

    raw_preset = presets[preset_key]
    if not isinstance(raw_preset, dict):
        raise ValueError(f"Preset '{preset_key}' must be a mapping")

    if preset_key in resolution_stack:
        cycle = " -> ".join((*resolution_stack, preset_key))
        raise ValueError(f"Preset inheritance cycle detected: {cycle}")

    preset_values = dict(raw_preset)
    parent_name = preset_values.pop("extends", None)
    if parent_name is None:
        return preset_values
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ValueError(f"Preset '{preset_key}' has invalid extends value: {parent_name!r}")
    if parent_name not in presets:
        raise ValueError(f"Preset '{preset_key}' extends unknown preset '{parent_name}'")

    resolved_parent = _resolve_preset_values(
        presets,
        parent_name,
        resolution_stack=(*resolution_stack, preset_key),
    )
    resolved_parent.update(preset_values)
    return resolved_parent


def _resolve_path_like(value: str | None, base_path: Path = REPO_ROOT) -> str | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_path / p).resolve()
    return str(p)


def _resolve_dataset_dir_path(value: str) -> str:
    p = Path(value).expanduser()
    if not p.is_absolute():
        return str((REPO_ROOT / p).resolve())

    resolved = p.resolve()
    if resolved.exists():
        return str(resolved)

    try:
        relative = resolved.relative_to(Path("/datasets/nv-ingest"))
    except ValueError:
        return str(resolved)

    user = os.environ.get("USER")
    if not user:
        return str(resolved)

    alternate = (Path("/raid") / user / relative).resolve()
    if alternate.exists():
        return str(alternate)

    return str(resolved)


def _resolve_query_csv_path(value: str | None, *, config_path: Path) -> str | None:
    if value is None:
        return None

    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p.resolve())

    resolved_candidates = [(base / p).resolve() for base in (config_path.parent, REPO_ROOT)]
    for candidate in resolved_candidates:
        if candidate.exists():
            return str(candidate)

    return str(resolved_candidates[0])


def _apply_env_overrides(config_dict: dict[str, Any]) -> None:
    env_map: dict[str, tuple[str, Any]] = {
        "HARNESS_DATASET": ("dataset", str),
        "HARNESS_DATASET_DIR": ("dataset_dir", str),
        "HARNESS_PRESET": ("preset", str),
        "HARNESS_QUERY_CSV": ("query_csv", str),
        "HARNESS_INPUT_TYPE": ("input_type", str),
        "HARNESS_RECALL_REQUIRED": ("recall_required", _parse_bool),
        "HARNESS_RECALL_MATCH_MODE": ("recall_match_mode", str),
        "HARNESS_RECALL_ADAPTER": ("recall_adapter", str),
        "HARNESS_ARTIFACTS_DIR": ("artifacts_dir", str),
        "HARNESS_RAY_ADDRESS": ("ray_address", str),
        "HARNESS_LANCEDB_URI": ("lancedb_uri", str),
        "HARNESS_HYBRID": ("hybrid", _parse_bool),
        "HARNESS_EMBED_MODEL_NAME": ("embed_model_name", str),
        "HARNESS_WRITE_DETECTION_FILE": ("write_detection_file", _parse_bool),
    }

    for key in TUNING_FIELDS:
        parser = _parse_auto_worker_value if key in AUTO_WORKER_FIELDS else _parse_number
        env_map[f"HARNESS_{key.upper()}"] = (key, parser)

    for env_key, (cfg_key, parser) in env_map.items():
        raw = os.getenv(env_key)
        if raw is None or raw == "":
            continue
        config_dict[cfg_key] = parser(raw)


def _parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            raise ValueError(f"Invalid override key in: {item}")

        low = raw_val.lower()
        if low in {"true", "false"}:
            parsed[key] = _parse_bool(raw_val)
        else:
            try:
                parsed[key] = _parse_number(raw_val)
            except ValueError:
                parsed[key] = raw_val
    return parsed


def _normalize_tuning_values(config_dict: dict[str, Any]) -> None:
    for name in AUTO_WORKER_FIELDS:
        if name not in config_dict:
            continue

        value = config_dict[name]
        if value is None:
            config_dict[name] = None
            continue

        if isinstance(value, str):
            raw = value.strip()
            if raw.lower() == "auto" or raw == "":
                config_dict[name] = None
                continue
            value = int(raw)

        normalized = int(value)
        config_dict[name] = None if normalized == 0 else normalized


def load_harness_config(
    *,
    config_file: str | None = None,
    dataset: str | None = None,
    preset: str | None = None,
    sweep_overrides: dict[str, Any] | None = None,
    cli_overrides: list[str] | None = None,
    cli_recall_required: bool | None = None,
) -> HarnessConfig:
    config_path = _resolve_config_path(config_file, DEFAULT_TEST_CONFIG_PATH)
    yaml_cfg = _read_yaml_mapping(config_path)

    active = dict(yaml_cfg.get("active", {}))
    datasets = dict(yaml_cfg.get("datasets", {}))
    presets = dict(yaml_cfg.get("presets", {}))

    sweep_data = dict(sweep_overrides or {})
    cli_override_map = _parse_cli_overrides(cli_overrides)

    dataset_ref = active.get("dataset")
    if "dataset" in sweep_data:
        dataset_ref = sweep_data["dataset"]
    if dataset is not None:
        dataset_ref = dataset
    if os.getenv("HARNESS_DATASET"):
        dataset_ref = os.environ["HARNESS_DATASET"]

    preset_ref = active.get("preset", "single_gpu")
    if "preset" in sweep_data:
        preset_ref = sweep_data["preset"]
    if preset is not None:
        preset_ref = preset
    if os.getenv("HARNESS_PRESET"):
        preset_ref = os.environ["HARNESS_PRESET"]

    merged: dict[str, Any] = dict(active)
    merged["preset"] = preset_ref

    dataset_label: str | None = None
    if dataset_ref:
        if dataset_ref in datasets:
            dataset_label = str(dataset_ref)
            dataset_cfg = dict(datasets[dataset_ref])
            path_val = dataset_cfg.pop("path", None)
            if path_val is not None:
                merged["dataset_dir"] = str(path_val)
            merged.update(dataset_cfg)
        else:
            dataset_label = Path(str(dataset_ref)).name
            merged["dataset_dir"] = str(dataset_ref)

    preset_values = _resolve_preset_values(presets, str(preset_ref))
    merged.update(preset_values)
    merged.update({k: v for k, v in sweep_data.items() if k not in {"dataset", "preset"}})
    merged.update(cli_override_map)
    if cli_recall_required is not None:
        merged["recall_required"] = cli_recall_required
    _apply_env_overrides(merged)
    _normalize_tuning_values(merged)

    dataset_dir = merged.get("dataset_dir")
    if dataset_dir is None:
        raise ValueError("dataset is required via active.dataset, --dataset, or sweep run")
    merged["dataset_dir"] = _resolve_dataset_dir_path(str(dataset_dir))
    merged["query_csv"] = _resolve_query_csv_path(merged.get("query_csv"), config_path=config_path)

    if merged.get("artifacts_dir") is not None:
        merged["artifacts_dir"] = _resolve_path_like(str(merged["artifacts_dir"]), REPO_ROOT)

    if merged.get("lancedb_uri") is None:
        merged["lancedb_uri"] = "lancedb"

    merged["dataset_label"] = dataset_label or Path(str(merged["dataset_dir"])).name
    merged["preset"] = str(merged.get("preset") or "single_gpu")

    if "query_csv" not in merged:
        merged["query_csv"] = None

    cfg = HarnessConfig(**{k: v for k, v in merged.items() if k in HarnessConfig.__dataclass_fields__})
    errors = cfg.validate()
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
    return cfg


def _load_nightly_runs_from_mapping(yaml_cfg: dict[str, Any], config_path: Path) -> list[dict[str, Any]]:
    runs = yaml_cfg.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"'runs' must be a list in {config_path}")
    normalized: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Run entry at index {idx} must be a mapping")
        if "dataset" not in run:
            raise ValueError(f"Run entry at index {idx} missing required key: dataset")
        normalized.append(dict(run))
    return normalized


def _normalize_nightly_slack_config(raw_cfg: Any, config_path: Path) -> dict[str, Any]:
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"'slack' must be a mapping in {config_path}")

    metric_keys = raw_cfg.get("metric_keys")
    if metric_keys is None:
        normalized_metric_keys = list(DEFAULT_NIGHTLY_SLACK_METRIC_KEYS)
    else:
        if not isinstance(metric_keys, list) or any(
            not isinstance(item, str) or not item.strip() for item in metric_keys
        ):
            raise ValueError(f"'slack.metric_keys' must be a list of non-empty strings in {config_path}")
        normalized_metric_keys = [item.strip() for item in metric_keys]

    title = raw_cfg.get("title")
    if title is None:
        normalized_title = "nemo_retriever Nightly Harness"
    else:
        normalized_title = str(title).strip()
        if not normalized_title:
            raise ValueError(f"'slack.title' must be a non-empty string in {config_path}")

    return {
        "enabled": bool(raw_cfg.get("enabled", True)),
        "title": normalized_title,
        "post_artifact_paths": bool(raw_cfg.get("post_artifact_paths", True)),
        "metric_keys": normalized_metric_keys,
    }


def load_nightly_config(config_file: str | None = None) -> dict[str, Any]:
    config_path = _resolve_config_path(config_file, DEFAULT_NIGHTLY_CONFIG_PATH)
    yaml_cfg = _read_yaml_mapping(config_path)
    preset = yaml_cfg.get("preset")
    if preset is not None:
        preset = str(preset).strip()
        if not preset:
            raise ValueError(f"'preset' must be a non-empty string in {config_path}")
    return {
        "config_path": str(config_path.resolve()),
        "preset": preset,
        "runs": _load_nightly_runs_from_mapping(yaml_cfg, config_path),
        "slack": _normalize_nightly_slack_config(yaml_cfg.get("slack", {}), config_path),
    }


def load_runs_config(config_file: str | None = None) -> list[dict[str, Any]]:
    return load_nightly_config(config_file)["runs"]
