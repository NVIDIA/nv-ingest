"""Docker Compose service manager implementation."""

import subprocess
import time
import urllib.request
from pathlib import Path

from .base import ServiceManager


class DockerComposeManager(ServiceManager):
    """Manages services using Docker Compose."""

    def __init__(self, config, repo_root: Path):
        """
        Initialize Docker Compose manager.

        Args:
            config: Configuration object with profiles and settings
            repo_root: Path to the repository root
        """
        super().__init__(config, repo_root)
        self.compose_file = str(repo_root / "docker-compose.yaml")

    def start(self, no_build: bool = False) -> int:
        """
        Start Docker Compose services with profiles.

        Args:
            no_build: If True, skip building Docker images

        Returns:
            0 on success, non-zero on failure
        """
        profile_list = self.config.profiles
        if not profile_list:
            print("No profiles specified")
            return 1

        # Build command with all profiles
        cmd = ["docker", "compose", "-f", self.compose_file, "--profile"]
        cmd += [profile_list[0]]
        for p in profile_list[1:]:
            cmd += ["--profile", p]

        # Add up command with or without build
        if not no_build:
            cmd += ["up", "--build", "-d"]
        else:
            cmd += ["up", "-d"]

        print("$", " ".join(cmd))
        return subprocess.call(cmd)

    def stop(self, clean: bool = False) -> int:
        """
        Stop Docker Compose services.

        Args:
            clean: If True, also remove volumes and orphans

        Returns:
            0 on success
        """
        print("Performing Docker Compose cleanup...")

        # Stop all services
        down_cmd = ["docker", "compose", "-f", self.compose_file, "--profile", "*", "down"]
        
        # Add cleanup flags if clean mode
        if clean:
            down_cmd += ["-v", "--remove-orphans"]
        
        print("$", " ".join(down_cmd))
        rc = subprocess.call(down_cmd)
        if rc != 0:
            print(f"Warning: docker compose down returned {rc}")

        # Remove containers forcefully
        rm_cmd = ["docker", "compose", "-f", self.compose_file, "--profile", "*", "rm", "--force"]
        print("$", " ".join(rm_cmd))
        rc2 = subprocess.call(rm_cmd)
        if rc2 != 0:
            print(f"Warning: docker compose rm returned {rc2}")

        return 0

    def check_readiness(self, timeout_s: int, check_milvus: bool = True) -> bool:
        """
        Poll the health endpoint until ready.

        Args:
            timeout_s: Timeout in seconds
            check_milvus: If True, also check Milvus health endpoint

        Returns:
            True if ready, False on timeout
        """
        url = self.get_service_url("health")
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            try:
                # Check main service health
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        # If Milvus check is enabled, verify it's also ready
                        if check_milvus:
                            hostname = getattr(self.config, "hostname", "localhost")
                            milvus_url = f"http://{hostname}:9091/healthz"
                            try:
                                with urllib.request.urlopen(milvus_url, timeout=5) as milvus_resp:
                                    if milvus_resp.status == 200:
                                        return True
                            except Exception:
                                pass
                        else:
                            return True
            except Exception:
                pass
            time.sleep(3)
        return False

    def get_service_url(self, service: str = "api") -> str:
        """
        Get service URL for Docker Compose (localhost).

        Args:
            service: Service name ("api" or "health")

        Returns:
            URL string
        """
        hostname = getattr(self.config, "hostname", "localhost")
        if service == "health":
            return f"http://{hostname}:7670/v1/health/ready"
        return f"http://{hostname}:7670"
