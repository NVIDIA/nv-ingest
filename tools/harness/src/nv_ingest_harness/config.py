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
    profiles: List[str] = field(default_factory=lambda: ["retrieval", "reranker"])  # Docker Compose only

    # Deployment configuration
    deployment_type: str = "compose"  # Options: compose, helm

    # Helm-specific configuration
    helm_bin: str = "helm"  # Helm binary command (e.g., "helm", "microk8s helm", "k3s helm")
    helm_sudo: bool = False  # Prepend sudo to helm commands (useful for microk8s, k3s)
    kubectl_bin: str = "kubectl"  # kubectl binary command (e.g., "kubectl", "microk8s kubectl")
    kubectl_sudo: Optional[bool] = None  # Prepend sudo to kubectl commands (defaults to helm_sudo if not set)
    helm_chart: Optional[str] = None  # Remote chart reference (e.g., "nim-nvstaging/nv-ingest"), None = use local chart
    helm_chart_version: Optional[str] = None  # Chart version (e.g., "26.1.0-RC7")
    helm_release: str = "nv-ingest"
    helm_namespace: str = "nv-ingest"
    helm_values_file: Optional[str] = None
    helm_values: Optional[dict] = None
    helm_port_forwards: Optional[List[dict]] = None  # List of {service, local_port, remote_port}

    # Runtime configuration
    sparse: bool = False
    gpu_search: bool = False
    embedding_model: str = "auto"
    llm_summarization_model: str = "nvdev/nvidia/llama-3.1-nemotron-70b-instruct"
    vdb_backend: str = "lancedb"
    hybrid: bool = False

    # Extraction configuration
    extract_text: bool = True
    extract_tables: bool = True
    extract_charts: bool = True
    extract_images: bool = False
    extract_infographics: bool = True
    extract_page_as_image: bool = False
    extract_method: Optional[str] = None
    text_depth: str = "page"
    table_output_format: str = "markdown"
    image_elements_modality: Optional[str] = None

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
    enable_beir: bool = False  # Enable BEIR metrics (NDCG, MAP, Precision)
    language_filter: Optional[str] = None  # Filter queries by language (e.g., "english")

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

        # Check deployment_type is valid
        if self.deployment_type not in ["compose", "helm"]:
            errors.append(f"deployment_type must be 'compose' or 'helm', got '{self.deployment_type}'")

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
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}\n")

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    # Start with active config from YAML
    config_dict = yaml_data.get("active", {}).copy()

    if not config_dict:
        raise ValueError("Config file must have 'active' section")

    # Flatten nested deployment configurations (compose and helm)
    # This allows cleaner YAML organization while keeping flat config structure
    if "compose" in config_dict:
        compose_config = config_dict.pop("compose")
        if isinstance(compose_config, dict):
            # Keys from compose section are used as-is (e.g., profiles)
            for key, value in compose_config.items():
                config_dict[key] = value

    if "helm" in config_dict:
        helm_config = config_dict.pop("helm")
        if isinstance(helm_config, dict):
            # Map helm section keys to their config field names
            for key, value in helm_config.items():
                # Keys that already have a prefix (kubectl_*) don't need helm_ prefix
                if key.startswith("kubectl_"):
                    config_dict[key] = value
                else:
                    # Other keys get helm_ prefix
                    config_dict[f"helm_{key}"] = value

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
        "DEPLOYMENT_TYPE": ("deployment_type", str),
        "HELM_BIN": ("helm_bin", str),
        "HELM_SUDO": ("helm_sudo", parse_bool),
        "KUBECTL_BIN": ("kubectl_bin", str),
        "KUBECTL_SUDO": ("kubectl_sudo", parse_bool),
        "HELM_CHART": ("helm_chart", str),
        "HELM_CHART_VERSION": ("helm_chart_version", str),
        "HELM_RELEASE": ("helm_release", str),
        "HELM_NAMESPACE": ("helm_namespace", str),
        "HELM_VALUES_FILE": ("helm_values_file", str),
        "SPARSE": ("sparse", parse_bool),
        "HYBRID": ("hybrid", parse_bool),
        "GPU_SEARCH": ("gpu_search", parse_bool),
        "EMBEDDING_NIM_MODEL_NAME": ("embedding_model", str),
        "LLM_SUMMARIZATION_MODEL": ("llm_summarization_model", str),
        "VDB_BACKEND": ("vdb_backend", str),
        "EXTRACT_TEXT": ("extract_text", parse_bool),
        "EXTRACT_TABLES": ("extract_tables", parse_bool),
        "EXTRACT_CHARTS": ("extract_charts", parse_bool),
        "EXTRACT_IMAGES": ("extract_images", parse_bool),
        "EXTRACT_INFOGRAPHICS": ("extract_infographics", parse_bool),
        "EXTRACT_PAGE_AS_IMAGE": ("extract_page_as_image", parse_bool),
        "EXTRACT_METHOD": ("extract_method", str),
        "TEXT_DEPTH": ("text_depth", str),
        "IMAGE_ELEMENTS_MODALITY": ("image_elements_modality", str),
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
        "ENABLE_BEIR": ("enable_beir", parse_bool),
        "LANGUAGE_FILTER": ("language_filter", str),
    }

    overrides = {}
    for env_var, (config_key, converter) in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None and value != "":
            converted = converter(value)
            if converted is not None:
                overrides[config_key] = converted

    return overrides


def expand_dataset_names(yaml_data: dict, dataset_input: str) -> List[str]:
    """
    Expand a dataset input string to a list of dataset names.

    Handles:
    - Single dataset name: "bo767" -> ["bo767"]
    - Comma-separated: "bo767,earnings" -> ["bo767", "earnings"]
    - Group name: "vidore" -> ["vidore_v3_finance_en", "vidore_v3_industrial", ...]
    - Mixed: "vidore_quick,bo767" -> ["vidore_v3_hr", "vidore_v3_industrial", "bo767"]

    Args:
        yaml_data: Parsed YAML data containing datasets and dataset_groups
        dataset_input: Raw dataset input string

    Returns:
        List of individual dataset names (expanded from groups)
    """
    dataset_groups = yaml_data.get("dataset_groups", {})

    raw_names = [name.strip() for name in dataset_input.split(",") if name.strip()]

    expanded = []
    for name in raw_names:
        if name in dataset_groups:
            expanded.extend(dataset_groups[name])
        else:
            expanded.append(name)

    seen = set()
    result = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            result.append(name)

    return result


def list_datasets(config_file: str = "test_configs.yaml") -> dict:
    """List available dataset shortcuts and groups"""
    config_path = Path(__file__).resolve().parents[2] / config_file

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    return {
        "datasets": yaml_data.get("datasets", {}),
        "groups": yaml_data.get("dataset_groups", {}),
    }


def list_presets(config_file: str = "test_configs.yaml") -> List[str]:
    """List available preset configurations"""
    config_path = Path(__file__).parent / config_file

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f)

    return list(yaml_data.get("presets", {}).keys())
