"""Helm service manager implementation."""

import subprocess
import time
import urllib.request
from pathlib import Path

from .base import ServiceManager


class HelmManager(ServiceManager):
    """Manages services using Helm."""

    def __init__(self, config, repo_root: Path):
        """
        Initialize Helm manager.

        Args:
            config: Configuration object with Helm settings
            repo_root: Path to the repository root
        """
        super().__init__(config, repo_root)
        self.chart_path = str(repo_root / "helm")
        self.release_name = getattr(config, "helm_release", "nv-ingest")
        self.namespace = getattr(config, "helm_namespace", "default")
        self.values_file = getattr(config, "helm_values_file", None)

    def start(self, no_build: bool = False) -> int:
        """
        Install or upgrade Helm release.

        Args:
            no_build: Not applicable for Helm (ignored)

        Returns:
            0 on success, non-zero on failure
        """
        cmd = [
            "helm",
            "upgrade",
            "--install",
            self.release_name,
            self.chart_path,
            "--namespace",
            self.namespace,
            "--create-namespace",
            "--wait",
            "--timeout",
            f"{self.config.readiness_timeout}s",
        ]

        # Add values file if specified
        if self.values_file:
            values_path = self.repo_root / self.values_file
            if values_path.exists():
                cmd += ["-f", str(values_path)]
            else:
                print(f"Warning: Values file {values_path} not found, skipping")

        # Add inline values from config (e.g., profiles mapped to enabled flags)
        if hasattr(self.config, "helm_values") and self.config.helm_values:
            for key, value in self.config.helm_values.items():
                cmd += ["--set", f"{key}={value}"]

        # Map profiles to Helm values if available
        if hasattr(self.config, "profiles") and self.config.profiles:
            for profile in self.config.profiles:
                # Enable corresponding services based on profile names
                cmd += ["--set", f"{profile}.enabled=true"]

        print("$", " ".join(cmd))
        return subprocess.call(cmd)

    def stop(self) -> int:
        """
        Uninstall Helm release.

        Returns:
            0 on success, non-zero on failure
        """
        print(f"Uninstalling Helm release {self.release_name}...")

        cmd = ["helm", "uninstall", self.release_name, "--namespace", self.namespace]

        print("$", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"Warning: helm uninstall returned {rc}")

        return 0

    def check_readiness(self, timeout_s: int) -> bool:
        """
        Check readiness by polling HTTP endpoint.

        Args:
            timeout_s: Timeout in seconds

        Returns:
            True if ready, False on timeout
        """
        url = self.get_service_url("health")
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(3)
        return False

    def get_service_url(self, service: str = "api") -> str:
        """
        Get service URL for Helm deployment.

        Args:
            service: Service name ("api" or "health")

        Returns:
            URL string
        """
        # Use hostname and port from config
        # For Helm deployments, this might be a LoadBalancer IP, NodePort, or port-forwarded localhost
        hostname = getattr(self.config, "hostname", "localhost")
        port = getattr(self.config, "service_port", 7670)

        if service == "health":
            return f"http://{hostname}:{port}/v1/health/ready"
        return f"http://{hostname}:{port}"
