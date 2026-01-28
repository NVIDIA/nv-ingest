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
    def stop(self, clean: bool = False) -> int:
        """
        Stop and cleanup services.

        Args:
            clean: If True, perform full cleanup (volumes, etc.)

        Returns:
            0 on success, non-zero on failure
        """
        pass

    @abstractmethod
    def check_readiness(self, timeout_s: int, check_milvus: bool = True, check_embedding: bool = True) -> bool:
        """
        Check if services are ready.

        Args:
            timeout_s: Timeout in seconds
            check_milvus: If True, also check Milvus health endpoint
            check_embedding: If True, also check embedding service health endpoint

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

    @abstractmethod
    def dump_logs(self, artifacts_dir: Path) -> int:
        """
        Dump logs of all managed containers/pods to artifacts directory.

        Args:
            artifacts_dir: Directory to write log files to

        Returns:
            0 on success, non-zero on failure
        """
        pass

    def restart(self, build: bool = False, clean: bool = True, timeout: int = 600) -> int:
        """
        Restart services (stop, clean, start, wait for readiness).

        Args:
            build: If True, rebuild images/charts
            clean: If True, perform full cleanup (volumes, etc.)
            timeout: Readiness timeout in seconds

        Returns:
            0 on success, non-zero on failure
        """
        print("Restarting services" + (" (with build)" if build else ""))

        # Stop and clean services
        rc = self.stop(clean=clean)
        if rc != 0:
            print(f"Warning: Service stop returned {rc}")

        # Start services
        rc = self.start(no_build=not build)
        if rc != 0:
            print(f"Failed to start services (exit code: {rc})")
            return rc

        # Wait for readiness
        if not self.check_readiness(timeout):
            print("Services failed to become ready within timeout")
            return 1

        print("Services restarted successfully!")
        return 0
