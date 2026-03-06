#!/usr/bin/env python3
"""
Validate consistency between Docker Compose and Helm deployment configurations.

This script compares image versions and environment variables for all services
defined in both docker-compose.yaml and helm/values.yaml, reporting any
discrepancies found.
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ServiceConfig:
    """Configuration for a deployable service."""

    name: str
    image_repo: Optional[str] = None
    image_tag: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConfigDiff:
    """Represents a configuration difference between deployments."""

    service_name: str
    field: str
    helm_value: Optional[str]
    compose_value: Optional[str]

    def __str__(self) -> str:
        return (
            f"  {self.service_name}.{self.field}:\n"
            f"    Helm:    {self.helm_value or '(not set)'}\n"
            f"    Compose: {self.compose_value or '(not set)'}"
        )


# Mapping between Docker Compose service names and Helm service names
# Most services are under nimOperator, but nv-ingest-ms-runtime is at the top level
SERVICE_MAPPING = {
    "page-elements": "page_elements",
    "graphic-elements": "graphic_elements",
    "table-structure": "table_structure",
    "ocr": "nemoretriever_ocr_v1",
    "embedding": "embedqa",
    "reranker": "llama_3_2_nv_rerankqa_1b_v2",
    "nemotron-parse": "nemotron_parse",
    "vlm": "nemotron_nano_12b_v2_vl",
    "audio": "audio",
    "nv-ingest-ms-runtime": "__MAIN__",  # Special marker for top-level config
}

# Environment variables to ignore in comparison (deployment-specific)
IGNORED_ENV_VARS = {
    "NGC_API_KEY",
    "NIM_NGC_API_KEY",
    "NVIDIA_API_KEY",
    "CUDA_VISIBLE_DEVICES",
    "NIM_ENABLE_OTEL",  # Often differs between deployments
    "NIM_OTEL_EXPORTER_OTLP_ENDPOINT",  # Endpoint URLs may differ
    "TRITON_OTEL_URL",  # Endpoint URLs may differ
    "OTEL_EXPORTER_OTLP_ENDPOINT",  # Main service OTEL endpoint
    "MINIO_ACCESS_KEY",  # Credentials
    "MINIO_SECRET_KEY",  # Credentials
    "MINIO_INTERNAL_ADDRESS",  # Deployment-specific DNS names
    "MINIO_PUBLIC_ADDRESS",  # Deployment-specific URLs
    "MESSAGE_CLIENT_HOST",  # DNS names differ (redis vs nv-ingest-redis-master)
    "EMBEDDING_NIM_ENDPOINT",  # DNS names differ (embedding:8000 vs llama-32-nv-embedqa-1b-v2:8000)
    "VLM_CAPTION_ENDPOINT",  # Conditionally set in Helm, may use external API
    "INGEST_RAY_LOG_LEVEL",  # May differ between dev/prod
    "RAY_num_server_call_thread",  # Ray config may differ
    "RAY_worker_num_grpc_internal_threads",  # Ray config may differ
    "PDF_SPLIT_PAGE_COUNT",  # Tuning parameter that may differ
    "REDIS_INGEST_TASK_QUEUE",  # Queue names may be deployment-specific
    "MILVUS_ENDPOINT",  # DNS names differ between deployments
    "IMAGE_STORAGE_PUBLIC_BASE_URL",  # Optional, may be empty
    "INGEST_EDGE_BUFFER_SIZE",  # May not be explicitly set in compose (uses default)
    # Service endpoint DNS names differ between Helm (full names) and Compose (short names)
    "YOLOX_GRPC_ENDPOINT",
    "YOLOX_HTTP_ENDPOINT",
    "YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT",
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
    "YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
    "OCR_GRPC_ENDPOINT",
    "OCR_HTTP_ENDPOINT",
}


def load_yaml(filepath: Path) -> dict:
    """Load and parse a YAML file."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def parse_compose_image(image_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse Docker Compose image string into repository and tag.

    Examples:
        'nvcr.io/nim/nvidia/nemoretriever-ocr-v1:1.2.1'
            -> ('nvcr.io/nim/nvidia/nemoretriever-ocr-v1', '1.2.1')
        '${OCR_IMAGE:-nvcr.io/nim/nvidia/nemoretriever-ocr-v1}:${OCR_TAG:-1.2.1}'
            -> ('nvcr.io/nim/nvidia/nemoretriever-ocr-v1', '1.2.1')
    """
    # Handle environment variable substitution patterns
    # Extract default values from ${VAR:-default} patterns
    image_str = re.sub(r"\$\{[^}]*:-([^}]+)\}", r"\1", image_str)
    image_str = re.sub(r"\$\{[^}]+\}", "", image_str)

    if ":" in image_str:
        parts = image_str.rsplit(":", 1)
        return parts[0].strip(), parts[1].strip()
    return image_str.strip(), None


def parse_compose_env(env_list: List) -> Dict[str, str]:
    """Parse Docker Compose environment variables."""
    env_dict = {}
    if not env_list:
        return env_dict

    for item in env_list:
        if isinstance(item, str):
            if "=" in item:
                key, value = item.split("=", 1)
                # Extract default values from ${VAR:-default} patterns
                value = re.sub(r"\$\{[^}]*:-([^}]+)\}", r"\1", value)
                value = re.sub(r"\$\{([^}]+)\}", r"${\1}", value)  # Keep unresolved vars
                env_dict[key.strip()] = value.strip()
        elif isinstance(item, dict):
            env_dict.update(item)

    return env_dict


def extract_compose_services(compose_config: dict) -> Dict[str, ServiceConfig]:
    """Extract service configurations from Docker Compose."""
    services = {}

    for service_name, service_def in compose_config.get("services", {}).items():
        if service_name not in SERVICE_MAPPING:
            continue  # Skip services not in our mapping

        image = service_def.get("image", "")
        repo, tag = parse_compose_image(image)

        environment = parse_compose_env(service_def.get("environment", []))

        services[service_name] = ServiceConfig(
            name=service_name,
            image_repo=repo,
            image_tag=tag,
            environment=environment,
        )

    return services


def parse_helm_env(env_list: List) -> Dict[str, str]:
    """Parse Helm environment variables."""
    env_dict = {}
    if not env_list:
        return env_dict

    for item in env_list:
        if isinstance(item, dict):
            name = item.get("name")
            value = item.get("value")
            if name:
                env_dict[name] = str(value) if value is not None else ""

    return env_dict


def extract_helm_services(helm_config: dict) -> Dict[str, ServiceConfig]:
    """Extract service configurations from Helm values."""
    services = {}

    # Handle main nv-ingest service (top-level configuration)
    main_image = helm_config.get("image", {})
    main_env_vars = helm_config.get("envVars", {})

    # Convert envVars dict to environment dict
    main_environment = {k: str(v) if v is not None else "" for k, v in main_env_vars.items()}

    services["nv-ingest-ms-runtime"] = ServiceConfig(
        name="nv-ingest-ms-runtime",
        image_repo=main_image.get("repository"),
        image_tag=str(main_image.get("tag")) if main_image.get("tag") else None,
        environment=main_environment,
    )

    # Handle NIM services under nimOperator
    nim_operator = helm_config.get("nimOperator", {})

    for helm_service_name, service_def in nim_operator.items():
        if helm_service_name in ["nimCache", "nimService"]:
            continue  # Skip configuration sections

        # Find corresponding compose service name
        compose_name = None
        for compose_key, helm_key in SERVICE_MAPPING.items():
            if helm_key == helm_service_name:
                compose_name = compose_key
                break

        if not compose_name:
            continue

        image_def = service_def.get("image", {})
        repo = image_def.get("repository")
        tag = image_def.get("tag")

        environment = parse_helm_env(service_def.get("env", []))

        services[compose_name] = ServiceConfig(
            name=compose_name,
            image_repo=repo,
            image_tag=str(tag) if tag else None,
            environment=environment,
        )

    return services


def normalize_value(value: Optional[str]) -> Optional[str]:
    """
    Normalize values for comparison.

    Handles:
    - Boolean normalization (true/True, false/False)
    - Numeric precision (0.8 vs 0.80)
    - Environment variable placeholders (${VAR:-})
    """
    if value is None or value == "":
        return None

    value = str(value).strip()

    # Handle empty environment variable defaults like ${VAR:-}
    if value.startswith("${") and value.endswith(":-}"):
        return None

    # Normalize booleans
    if value.lower() in ("true", "false"):
        return value.lower()

    # Try to normalize numeric values
    try:
        # Handle integers
        if "." not in value:
            int_val = int(value)
            return str(int_val)
        # Handle floats
        else:
            float_val = float(value)
            # Normalize to consistent precision
            return str(float_val)
    except (ValueError, TypeError):
        pass

    return value


def compare_services(
    compose_services: Dict[str, ServiceConfig], helm_services: Dict[str, ServiceConfig]
) -> List[ConfigDiff]:
    """Compare service configurations and return list of differences."""
    diffs = []

    # Get all service names from both configs
    all_services = set(compose_services.keys()) | set(helm_services.keys())

    for service_name in sorted(all_services):
        compose_svc = compose_services.get(service_name)
        helm_svc = helm_services.get(service_name)

        if not compose_svc or not helm_svc:
            # Service exists in only one config - this might be intentional (profiles)
            continue

        # Compare image repository
        if compose_svc.image_repo and helm_svc.image_repo:
            if compose_svc.image_repo != helm_svc.image_repo:
                diffs.append(
                    ConfigDiff(
                        service_name=service_name,
                        field="image_repository",
                        helm_value=helm_svc.image_repo,
                        compose_value=compose_svc.image_repo,
                    )
                )

        # Compare image tag
        if compose_svc.image_tag and helm_svc.image_tag:
            if compose_svc.image_tag != helm_svc.image_tag:
                diffs.append(
                    ConfigDiff(
                        service_name=service_name,
                        field="image_tag",
                        helm_value=helm_svc.image_tag,
                        compose_value=compose_svc.image_tag,
                    )
                )

        # Compare environment variables
        all_env_keys = set(compose_svc.environment.keys()) | set(helm_svc.environment.keys())

        for env_key in sorted(all_env_keys):
            if env_key in IGNORED_ENV_VARS:
                continue

            compose_val = compose_svc.environment.get(env_key)
            helm_val = helm_svc.environment.get(env_key)

            # Normalize values for comparison
            compose_val_normalized = normalize_value(compose_val)
            helm_val_normalized = normalize_value(helm_val)

            # Skip if both are unset or empty after normalization
            if not compose_val_normalized and not helm_val_normalized:
                continue

            # Check for differences
            if compose_val_normalized != helm_val_normalized:
                diffs.append(
                    ConfigDiff(
                        service_name=service_name,
                        field=f"env.{env_key}",
                        helm_value=helm_val,
                        compose_value=compose_val,
                    )
                )

    return diffs


def main() -> int:
    """Main validation logic."""
    # Determine repository root (script is in ci/scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    compose_path = repo_root / "docker-compose.yaml"
    helm_path = repo_root / "helm" / "values.yaml"

    # Load configurations
    print("Loading configuration files...")
    print(f"  Docker Compose: {compose_path}")
    print(f"  Helm values:    {helm_path}")
    print()

    try:
        compose_config = load_yaml(compose_path)
        helm_config = load_yaml(helm_path)
    except Exception as e:
        print(f"❌ Error loading configuration files: {e}", file=sys.stderr)
        return 1

    # Extract service configurations
    compose_services = extract_compose_services(compose_config)
    helm_services = extract_helm_services(helm_config)

    print(f"Found {len(compose_services)} Docker Compose services")
    print(f"Found {len(helm_services)} Helm services")
    print()

    # Compare configurations
    diffs = compare_services(compose_services, helm_services)

    if not diffs:
        print("✅ All service configurations are consistent!")
        return 0

    # Report differences
    print(f"❌ Found {len(diffs)} configuration discrepancies:\n")

    # Group diffs by service
    diffs_by_service: Dict[str, List[ConfigDiff]] = {}
    for diff in diffs:
        if diff.service_name not in diffs_by_service:
            diffs_by_service[diff.service_name] = []
        diffs_by_service[diff.service_name].append(diff)

    # Print grouped by service
    for service_name in sorted(diffs_by_service.keys()):
        service_diffs = diffs_by_service[service_name]
        print(f"Service: {service_name}")
        for diff in service_diffs:
            print(diff)
        print()

    print(f"Total discrepancies: {len(diffs)}")
    print("\nPlease update the configurations to ensure consistency between deployments.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
