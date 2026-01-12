"""Service manager factory and exports."""

from pathlib import Path

from .base import ServiceManager
from .docker_compose import DockerComposeManager
from .helm import HelmManager


def create_service_manager(config, repo_root: Path) -> ServiceManager:
    """
    Factory to create the appropriate service manager based on config.

    Args:
        config: Configuration object with deployment_type attribute
        repo_root: Path to the repository root

    Returns:
        ServiceManager instance (DockerComposeManager or HelmManager)

    Raises:
        ValueError: If deployment_type is unknown
    """
    deployment_type = getattr(config, "deployment_type", "docker-compose")

    if deployment_type == "docker-compose":
        return DockerComposeManager(config, repo_root)
    elif deployment_type == "helm":
        return HelmManager(config, repo_root)
    else:
        raise ValueError(f"Unknown deployment_type: {deployment_type}. Must be 'docker-compose' or 'helm'")


__all__ = ["ServiceManager", "create_service_manager", "DockerComposeManager", "HelmManager"]
