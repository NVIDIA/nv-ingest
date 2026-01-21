"""Base class for service orchestration managers."""

from abc import ABC, abstractmethod
from pathlib import Path


class ServiceManager(ABC):
    """Abstract base for service orchestration managers."""

    def __init__(self, config, repo_root: Path):
        """
        Initialize the service manager.

        Args:
            config: Configuration object with deployment settings
            repo_root: Path to the repository root
        """
        self.config = config
        self.repo_root = repo_root

    @abstractmethod
    def start(self, no_build: bool = False) -> int:
        """
        Start services.

        Args:
            no_build: If True, skip building images/charts

        Returns:
            0 on success, non-zero on failure
        """
        pass

    @abstractmethod
    def stop(self) -> int:
        """
        Stop and cleanup services.

        Returns:
            0 on success, non-zero on failure
        """
        pass

    @abstractmethod
    def check_readiness(self, timeout_s: int, check_milvus: bool = True) -> bool:
        """
        Check if services are ready.

        Args:
            timeout_s: Timeout in seconds
            check_milvus: If True, also check Milvus health endpoint

        Returns:
            True when ready, False on timeout
        """
        pass

    @abstractmethod
    def get_service_url(self, service: str = "api") -> str:
        """
        Get the URL for a service endpoint.

        Args:
            service: Service name (e.g., "api", "health")

        Returns:
            URL string
        """
        pass
