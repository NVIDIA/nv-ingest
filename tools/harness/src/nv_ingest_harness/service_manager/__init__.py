"""Service manager factory and exports."""

from pathlib import Path

from nv_ingest_harness.service_manager.base import ServiceManager
from nv_ingest_harness.service_manager.docker_compose import DockerComposeManager
from nv_ingest_harness.service_manager.helm import HelmManager


def create_service_manager(config, repo_root: Path, sku: str | None = None) -> ServiceManager:
    """
    Factory to create the appropriate service manager based on config.

    Args:
        config: Configuration object with deployment_type attribute
        repo_root: Path to the repository root
        sku: Optional GPU SKU for override file (Compose: docker-compose.<sku>.yaml;
            Helm: helm/overrides/values-<sku>.yaml)

    Returns:
        ServiceManager instance (DockerComposeManager or HelmManager)

    Raises:
        ValueError: If deployment_type is unknown
    """
    deployment_type = getattr(config, "deployment_type", "compose")

    if deployment_type == "compose":
        return DockerComposeManager(config, repo_root, sku=sku)
    elif deployment_type == "helm":
        return HelmManager(config, repo_root, sku=sku)
    else:
        raise ValueError(f"Unknown deployment_type: {deployment_type}. Must be 'compose' or 'helm'")


__all__ = ["ServiceManager", "create_service_manager", "DockerComposeManager", "HelmManager"]
