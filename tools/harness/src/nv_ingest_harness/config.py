"""
Configuration management for nv-ingest integration tests.

Loads test configuration from test_configs.yaml with support for:
- Direct YAML editing (primary workflow)
- Dataset shortcuts
- Environment variable overrides
- CLI argument overrides

Precedence: CLI args > Env vars > Dataset-specific config (path + extraction + recall_dataset) > YAML active config
"""

import os
import glob
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TestConfig:
    """Test configuration matching YAML active section"""

    # Dataset configuration
    dataset_dir: str
    test_name: Optional[str] = None

    # API configuration
    api_version: str = "v1"
    pdf_split_page_count: Optional[int] = None

    # Infrastructure
    hostname: str = "localhost"
    readiness_timeout: int = 600
    profiles: List[str] = field(default_factory=lambda: ["retrieval", "table-structure"])

    # Runtime configuration
    sparse: bool = True
    gpu_search: bool = False
    embedding_model: str = "auto"
    llm_summarization_model: str = "nvdev/nvidia/llama-3.1-nemotron-70b-instruct"
    vdb_backend: str = "milvus"

    # Extraction configuration
    extract_text: bool = True
    extract_tables: bool = True
    extract_charts: bool = True
    extract_images: bool = False
    extract_infographics: bool = True
    text_depth: str = "page"
    table_output_format: str = "markdown"

    # Optional pipeline steps
    enable_caption: bool = False
    caption_prompt: Optional[str] = None
    caption_reasoning: Optional[bool] = None
    enable_split: bool = False
    split_chunk_size: int = 1024
    split_chunk_overlap: int = 150
    enable_image_storage: bool = False  # Server-side image storage (MinIO/local disk)

    # Image storage configuration
    store_structured: bool = True
    store_images: bool = True
    storage_uri: Optional[str] = None  # file:///path or s3://bucket/path
    storage_options: Optional[dict] = None
    public_base_url: Optional[str] = None

    # Storage configuration
    spill_dir: str = "/tmp/spill"
    artifacts_dir: Optional[str] = None
    collection_name: Optional[str] = None
    lancedb_dir: Optional[str] = None

    # Recall configuration
    reranker_mode: str = "none"  # Options: "none", "with", "both"
    recall_top_k: int = 10
    ground_truth_dir: Optional[str] = None
    recall_dataset: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Check pdf_split_page_count requires api_version v2
        if self.pdf_split_page_count is not None:
            if self.api_version != "v2":
                errors.append(
                    f"pdf_split_page_count={self.pdf_split_page_count} requires api_version='v2', "
                    f"but got api_version='{self.api_version}'"
                )
            if not (1 <= self.pdf_split_page_count <= 128):
                errors.append(f"pdf_split_page_count must be between 1 and 128, got {self.pdf_split_page_count}")

        # Check text_depth is valid (from TextTypeEnum)
        valid_text_depths = ["block", "body", "document", "header", "line", "nearby_block", "other", "page", "span"]
        if self.text_depth not in valid_text_depths:
            errors.append(f"text_depth must be one of {valid_text_depths}, got '{self.text_depth}'")

        # Check table_output_format is valid (from TableFormatEnum)
        valid_table_formats = ["html", "image", "latex", "markdown", "pseudo_markdown", "simple"]
        if self.table_output_format not in valid_table_formats:
            errors.append(f"table_output_format must be one of {valid_table_formats}, got '{self.table_output_format}'")

        # Check api_version is valid
        if self.api_version not in ["v1", "v2"]:
            errors.append(f"api_version must be 'v1' or 'v2', got '{self.api_version}'")

        # Check reranker_mode is valid
        if self.reranker_mode not in ["none", "with", "both"]:
            errors.append(f"reranker_mode must be 'none', 'with', or 'both', got '{self.reranker_mode}'")

        # Check vdb_backend is valid
        if self.vdb_backend not in ["milvus", "lancedb"]:
            errors.append(f"vdb_backend must be 'milvus' or 'lancedb', got '{self.vdb_backend}'")

        # Check dataset_dir exists (can be file, directory, or glob pattern)
        # Check if it's a glob pattern (contains *, ?, or [)
        is_glob = any(char in self.dataset_dir for char in ["*", "?", "["])

        if is_glob:
            # For glob patterns, check if any files match
            matching_files = list(glob.glob(self.dataset_dir, recursive=True))
            if not matching_files:
                errors.append(f"glob pattern matches no files: {self.dataset_dir}")
        else:
            # For regular paths, check if it exists
            if not os.path.exists(self.dataset_dir):
                errors.append(f"dataset path does not exist: {self.dataset_dir}")
            elif not (os.path.isfile(self.dataset_dir) or os.path.isdir(self.dataset_dir)):
                errors.append(f"dataset path must be a file or directory: {self.dataset_dir}")

        return errors


def _get_dataset_config(yaml_data: dict, dataset_name: str) -> dict:
    """
    Get complete dataset configuration including path and extraction settings.

    Args:
        yaml_data: Parsed YAML data
        dataset_name: Dataset shortcut name

    Returns:
        Dictionary with dataset path and extraction config, or empty dict if not found
    """
    datasets = yaml_data.get("datasets", {})
    dataset_config = datasets.get(dataset_name, {})

    # Handle backward compatibility: if datasets is a simple dict (name -> path)
    # convert to new format
    if isinstance(dataset_config, str):
        return {"path": dataset_config}

    return dataset_config


def load_config(config_file: str = "test_configs.yaml", case: Optional[str] = None, **cli_overrides) -> TestConfig:
    """
    Load test configuration from YAML with overrides.

    Precedence: CLI args > Env vars > Dataset-specific config > YAML active config

    Args:
        config_file: Path to YAML config file (relative to this script)
        case: Test case name (used to determine if recall section should be merged)
        **cli_overrides: CLI argument overrides (e.g., dataset="bo767", api_version="v2")

    Returns:
        TestConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration validation fails
    """
    config_path = Path(__file__).resolve().parents[2] / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}\n")

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    # Start with active config from YAML
    config_dict = yaml_data.get("active", {}).copy()

    if not config_dict:
        raise ValueError("Config file must have 'active' section")

    # Merge recall section when running recall test cases
    # The recall section provides additional configuration for recall evaluation
    if case in ("recall", "e2e_recall"):
        recall_section = yaml_data.get("recall", {})
        if recall_section:
            # Merge recall section (recall section overrides active section for conflicts)
            config_dict.update(recall_section)

    # Handle dataset shortcuts and apply dataset-specific extraction configs
    if "dataset" in cli_overrides:
        dataset_name = cli_overrides.pop("dataset")
        if dataset_name is not None:  # Only override if actually provided
            dataset_config = _get_dataset_config(yaml_data, dataset_name)

            if dataset_config and "path" in dataset_config:
                # Configured dataset: extract path and apply extraction settings
                config_dict["dataset_dir"] = dataset_config["path"]

                # Apply dataset-specific configs (extraction settings + recall_dataset, excluding path)
                dataset_specific_config = {k: v for k, v in dataset_config.items() if k != "path"}
                if dataset_specific_config:
                    config_dict.update(dataset_specific_config)
            else:
                # Not a configured dataset, treat dataset_name as direct path
                config_dict["dataset_dir"] = dataset_name

    # Apply environment variable overrides
    env_overrides = _load_env_overrides()
    config_dict.update(env_overrides)

    # Apply CLI overrides (highest priority)
    # Convert kebab-case CLI args to snake_case config keys
    normalized_cli = {k.replace("-", "_"): v for k, v in cli_overrides.items() if v is not None}
    config_dict.update(normalized_cli)

    # Build config object
    try:
        config = TestConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    return config


def _load_env_overrides() -> dict:
    """Load configuration overrides from environment variables"""

    def parse_bool(value: str) -> bool:
        """Parse boolean from string"""
        return value.lower() in ("true", "1", "yes")

    def parse_int(value: str) -> int:
        """Parse integer from string"""
        try:
            return int(value)
        except ValueError:
            return None

    def parse_list(value: str) -> List[str]:
        """Parse comma-separated list"""
        return [item.strip() for item in value.split(",") if item.strip()]

    # Map environment variables to config keys with optional converters
    env_mapping = {
        "DATASET_DIR": ("dataset_dir", str),
        "TEST_NAME": ("test_name", str),
        "API_VERSION": ("api_version", str),
        "PDF_SPLIT_PAGE_COUNT": ("pdf_split_page_count", parse_int),
        "HOSTNAME": ("hostname", str),
        "READINESS_TIMEOUT": ("readiness_timeout", parse_int),
        "PROFILES": ("profiles", parse_list),
        "SPARSE": ("sparse", parse_bool),
        "GPU_SEARCH": ("gpu_search", parse_bool),
        "EMBEDDING_NIM_MODEL_NAME": ("embedding_model", str),
        "LLM_SUMMARIZATION_MODEL": ("llm_summarization_model", str),
        "VDB_BACKEND": ("vdb_backend", str),
        "EXTRACT_TEXT": ("extract_text", parse_bool),
        "EXTRACT_TABLES": ("extract_tables", parse_bool),
        "EXTRACT_CHARTS": ("extract_charts", parse_bool),
        "EXTRACT_IMAGES": ("extract_images", parse_bool),
        "EXTRACT_INFOGRAPHICS": ("extract_infographics", parse_bool),
        "TEXT_DEPTH": ("text_depth", str),
        "TABLE_OUTPUT_FORMAT": ("table_output_format", str),
        "ENABLE_CAPTION": ("enable_caption", parse_bool),
        "CAPTION_PROMPT": ("caption_prompt", str),
        "CAPTION_REASONING": ("caption_reasoning", parse_bool),
        "ENABLE_SPLIT": ("enable_split", parse_bool),
        "SPLIT_CHUNK_SIZE": ("split_chunk_size", parse_int),
        "SPLIT_CHUNK_OVERLAP": ("split_chunk_overlap", parse_int),
        "ENABLE_IMAGE_STORAGE": ("enable_image_storage", parse_bool),
        "SPILL_DIR": ("spill_dir", str),
        "ARTIFACTS_DIR": ("artifacts_dir", str),
        "COLLECTION_NAME": ("collection_name", str),
        "LANCEDB_DIR": ("lancedb_dir", str),
        "RERANKER_MODE": ("reranker_mode", str),
        "RECALL_TOP_K": ("recall_top_k", parse_int),
        "GROUND_TRUTH_DIR": ("ground_truth_dir", str),
        "RECALL_DATASET": ("recall_dataset", str),
    }

    overrides = {}
    for env_var, (config_key, converter) in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None and value != "":
            converted = converter(value)
            if converted is not None:
                overrides[config_key] = converted

    return overrides


def list_datasets(config_file: str = "test_configs.yaml") -> dict:
    """List available dataset shortcuts"""
    config_path = Path(__file__).parent / config_file

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    return yaml_data.get("datasets", {})


def list_presets(config_file: str = "test_configs.yaml") -> List[str]:
    """List available preset configurations"""
    config_path = Path(__file__).parent / config_file

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    return list(yaml_data.get("presets", {}).keys())
