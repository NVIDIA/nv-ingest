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
        # Helm binary command (supports "helm", "microk8s helm", "k3s helm", etc.)
        helm_bin = getattr(config, "helm_bin", "helm")
        helm_sudo = getattr(config, "helm_sudo", False)

        # Build base command with optional sudo prefix
        if helm_sudo:
            self.helm_cmd = ["sudo"] + helm_bin.split()
        else:
            self.helm_cmd = helm_bin.split()  # Split to support multi-word commands

        # Use remote chart if specified, otherwise fall back to local chart
        self.chart_ref = getattr(config, "helm_chart", None)
        if not self.chart_ref:
            # Default to local chart path if no remote chart specified
            self.chart_ref = str(repo_root / "helm")
        self.chart_version = getattr(config, "helm_chart_version", None)
        self.release_name = getattr(config, "helm_release", "nv-ingest")
        self.namespace = getattr(config, "helm_namespace", "nv-ingest")
        self.values_file = getattr(config, "helm_values_file", None)
        self.service_port = getattr(config, "service_port", 7670)

        # Port forwarding processes (list of tuples: (process, description))
        self.port_forward_processes: list[tuple[subprocess.Popen, str]] = []

        # kubectl command (for port forwarding)
        kubectl_bin = getattr(config, "kubectl_bin", "kubectl")
        kubectl_sudo = getattr(config, "kubectl_sudo") or helm_sudo  # Default to same as helm_sudo

        if kubectl_sudo:
            self.kubectl_cmd = ["sudo"] + kubectl_bin.split()
        else:
            self.kubectl_cmd = kubectl_bin.split()

    def start(self, no_build: bool = False) -> int:
        """
        Install or upgrade Helm release.

        Args:
            no_build: Not applicable for Helm (ignored)

        Returns:
            0 on success, non-zero on failure
        """
        cmd = self.helm_cmd + [
            "upgrade",
            "--install",
            self.release_name,
            self.chart_ref,
            "--namespace",
            self.namespace,
            "--create-namespace",
        ]

        # Add version if specified (only valid for remote charts)
        if self.chart_version:
            cmd += ["--version", self.chart_version]

        # Add values file if specified
        if self.values_file:
            values_path = self.repo_root / self.values_file
            if values_path.exists():
                cmd += ["-f", str(values_path)]
            else:
                print(f"Warning: Values file {values_path} not found, skipping")

        # Add inline values from config
        if hasattr(self.config, "helm_values") and self.config.helm_values:
            for key, value in self.config.helm_values.items():
                cmd += ["--set", f"{key}={value}"]

        print("$", " ".join(cmd))
        rc = subprocess.call(cmd)

        if rc == 0:
            # Start port forwarding for all configured services
            self._start_port_forwards()

        return rc

    def _find_services_by_pattern(self, pattern: str) -> list[str]:
        """
        Find services matching a pattern (supports wildcards).

        Args:
            pattern: Service name or pattern (e.g., "nv-ingest", "*embed*")

        Returns:
            List of matching service names
        """
        # If no wildcards, return as-is
        if "*" not in pattern:
            return [pattern]

        # Query kubectl for services in the namespace
        cmd = self.kubectl_cmd + ["get", "services", "-n", self.namespace, "-o", "name"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"Warning: Could not list services: {result.stderr.strip()}")
                return []

            # Parse output (format: "service/name")
            service_names = [line.split("/")[-1] for line in result.stdout.strip().split("\n") if line]

            # Convert pattern to regex-like matching
            import re

            regex_pattern = pattern.replace("*", ".*")
            matches = [svc for svc in service_names if re.search(regex_pattern, svc)]

            return matches
        except Exception as e:
            print(f"Warning: Error finding services: {e}")
            return []

    def _start_port_forwards(self) -> None:
        """Start port forwarding for all configured services."""
        # Get port forward configuration
        port_forwards = getattr(self.config, "helm_port_forwards", None)

        if not port_forwards:
            # Default: forward main service only
            port_forwards = [
                {
                    "service": self.release_name,
                    "local_port": self.service_port,
                    "remote_port": self.service_port,
                }
            ]

        # Start each port forward
        for pf_config in port_forwards:
            service_pattern = pf_config.get("service")
            local_port = pf_config.get("local_port")
            remote_port = pf_config.get("remote_port")

            if not all([service_pattern, local_port, remote_port]):
                print(f"Warning: Invalid port forward config: {pf_config}")
                continue

            # Find matching services
            service_names = self._find_services_by_pattern(service_pattern)

            if not service_names:
                print(f"Warning: No services found matching pattern '{service_pattern}'")
                continue

            # Start port forward for each matching service
            for service_name in service_names:
                self._start_single_port_forward(
                    service_name=service_name,
                    local_port=local_port,
                    remote_port=remote_port,
                )

    def _start_single_port_forward(
        self, service_name: str, local_port: int, remote_port: int, timeout: int = 120, retry_interval: int = 5
    ) -> None:
        """
        Start kubectl port-forward for a single service with retry logic.

        Args:
            service_name: Name of the service to forward
            local_port: Local port to bind to
            remote_port: Remote port on the service
            timeout: Maximum time to wait for port-forward to succeed (seconds)
            retry_interval: Time between retry attempts (seconds)
        """
        # Build port-forward command
        # Format: kubectl port-forward -n namespace service/name local_port:remote_port
        cmd = self.kubectl_cmd + [
            "port-forward",
            "-n",
            self.namespace,
            f"service/{service_name}",
            f"{local_port}:{remote_port}",
        ]

        description = f"{service_name} ({local_port}:{remote_port})"
        print("$", " ".join(cmd), "(background)")
        print(f"Waiting for {service_name} pod to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        attempt = 1

        while time.time() - start_time < timeout:
            try:
                # Start port-forward in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Give it a moment to establish
                time.sleep(2)

                # Check if process is still running (didn't exit with error)
                poll_result = process.poll()
                if poll_result is not None:
                    # Process exited, read error output
                    _, stderr = process.communicate()
                    error_msg = stderr.decode("utf-8").strip() if stderr else f"exit code {poll_result}"

                    # Check if it's a "pod not running" error that we should retry
                    if "pod is not running" in error_msg or "Pending" in error_msg:
                        elapsed = int(time.time() - start_time)
                        print(f"  Attempt {attempt}: Pod not ready yet (elapsed: {elapsed}s)")
                        time.sleep(retry_interval)
                        attempt += 1
                        continue
                    else:
                        # Different error, don't retry
                        print(f"Error: Port forwarding for {service_name} failed: {error_msg}")
                        return
                else:
                    # Success! Process is still running
                    self.port_forward_processes.append((process, description))
                    print(f"Port forwarding started for {description} (PID: {process.pid})")
                    return

            except Exception as e:
                print(f"Error: Failed to start port forwarding for {service_name}: {e}")
                return

        # Timeout reached
        print(f"Error: Port forwarding for {service_name} failed to establish after {timeout}s (pod may not be ready)")

    def _stop_port_forwards(self) -> None:
        """Stop all port-forward processes."""
        if not self.port_forward_processes:
            return

        print(f"Stopping {len(self.port_forward_processes)} port forward(s)...")

        for process, description in self.port_forward_processes:
            try:
                print(f"  Stopping {description} (PID: {process.pid})...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("    Port forward didn't terminate, killing...")
                process.kill()
            except Exception as e:
                print(f"    Warning: Error stopping port forward: {e}")

        self.port_forward_processes = []

    def print_port_forward_commands(self) -> None:
        """Print commands to manually recreate port forwards."""
        port_forwards = getattr(self.config, "helm_port_forwards", None)
        if not port_forwards:
            port_forwards = [
                {
                    "service": self.release_name,
                    "local_port": self.service_port,
                    "remote_port": self.service_port,
                }
            ]

        print("\nTo manually recreate port forwards, run:")
        print("=" * 60)
        for pf in port_forwards:
            service = pf.get("service")
            local = pf.get("local_port")
            remote = pf.get("remote_port")

            if "*" in service:
                print(f"  # For pattern: {service}")
                matches = self._find_services_by_pattern(service)
                for match in matches:
                    cmd = " ".join(
                        self.kubectl_cmd
                        + ["port-forward", "-n", self.namespace, f"service/{match}", f"{local}:{remote}"]
                    )
                    print(f"  {cmd} &")
            else:
                cmd = " ".join(
                    self.kubectl_cmd + ["port-forward", "-n", self.namespace, f"service/{service}", f"{local}:{remote}"]
                )
                print(f"  {cmd} &")
        print("=" * 60)

    def stop(self) -> int:
        """
        Uninstall Helm release.

        Note: Port forwards should be stopped separately via _stop_port_forwards()
        to allow keeping services up while cleaning up port-forward processes.

        Returns:
            0 on success, non-zero on failure
        """
        print(f"Uninstalling Helm release {self.release_name}...")

        cmd = self.helm_cmd + ["uninstall", self.release_name, "--namespace", self.namespace]

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
