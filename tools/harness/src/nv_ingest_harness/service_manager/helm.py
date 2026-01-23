"""Helm service manager implementation."""

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from nv_ingest_harness.service_manager.base import ServiceManager
from nv_ingest_harness.utils.interact import run_cmd


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

        # Port forwarding processes (list of tuples: (process, description))
        self.port_forward_processes: list[tuple[subprocess.Popen, str]] = []

        # kubectl command (for port forwarding)
        kubectl_bin = getattr(config, "kubectl_bin", "kubectl")
        self.kubectl_sudo = getattr(config, "kubectl_sudo") or helm_sudo  # Default to same as helm_sudo

        if self.kubectl_sudo:
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

        rc = run_cmd(cmd)

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
            # Default: forward main service only (port 7670)
            port_forwards = [
                {
                    "service": self.release_name,
                    "local_port": 7670,
                    "remote_port": 7670,
                }
            ]

        # Group port forwards by service (after resolving patterns)
        # Map: service_name -> [(local_port, remote_port), ...]
        service_port_map = {}

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

            # Add ports to each matching service
            for service_name in service_names:
                if service_name not in service_port_map:
                    service_port_map[service_name] = []
                service_port_map[service_name].append((local_port, remote_port))

        # Start consolidated port forwards
        for service_name, port_pairs in service_port_map.items():
            self._start_single_port_forward(
                service_name=service_name,
                port_pairs=port_pairs,
            )

    def _start_single_port_forward(
        self, service_name: str, port_pairs: list[tuple[int, int]], timeout: int = 120, retry_interval: int = 5
    ) -> None:
        """
        Start kubectl port-forward for a single service with multiple ports and retry logic.

        The port-forward is wrapped in a resilient shell loop that automatically restarts
        if the connection drops (e.g., due to pod restarts).

        Args:
            service_name: Name of the service to forward
            port_pairs: List of (local_port, remote_port) tuples
            timeout: Maximum time to wait for port-forward to succeed (seconds)
            retry_interval: Time between retry attempts (seconds)
        """
        # Build base kubectl port-forward command
        kubectl_cmd_str = " ".join(self.kubectl_cmd)
        port_strs = [f"{local}:{remote}" for local, remote in port_pairs]
        ports_arg = " ".join(port_strs)

        # Create a resilient wrapper script that auto-restarts on failure
        # This handles pod restarts and transient failures
        wrapper_script = f"""
while true; do
    echo "[$(date)] Starting port-forward for {service_name}..." >&2
    {kubectl_cmd_str} port-forward -n {self.namespace} service/{service_name} {ports_arg}
    EXIT_CODE=$?
    echo "[$(date)] Port-forward exited with code $EXIT_CODE, restarting in {retry_interval}s..." >&2
    sleep {retry_interval}
done
"""

        # Build description
        ports_desc = " ".join(port_strs)
        description = f"{service_name} ({ports_desc})"
        print(
            f"$ {kubectl_cmd_str} port-forward -n {self.namespace} service/{service_name} "
            f"{ports_arg} (background, auto-restart)"
        )
        print(f"Waiting for {service_name} pod to be ready (timeout: {timeout}s)...")

        start_time = time.time()
        attempt = 1

        while time.time() - start_time < timeout:
            try:
                # Start resilient port-forward wrapper in background
                # Create new process group so we can kill all children later
                process = subprocess.Popen(
                    ["bash", "-c", wrapper_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,  # Create new process group
                )

                # Give it a moment to establish
                time.sleep(2)

                # Check if wrapper process is still running
                poll_result = process.poll()
                if poll_result is not None:
                    # Wrapper exited unexpectedly, read error output
                    _, stderr = process.communicate()
                    error_msg = stderr.decode("utf-8").strip() if stderr else f"exit code {poll_result}"

                    # Check if it's a "pod not running" error that we should retry
                    if "pod is not running" in error_msg or "Pending" in error_msg or "not found" in error_msg:
                        elapsed = int(time.time() - start_time)
                        print(f"  Attempt {attempt}: Pod not ready yet (elapsed: {elapsed}s)")
                        time.sleep(retry_interval)
                        attempt += 1
                        continue
                    else:
                        # Different error, don't retry initial setup
                        print(f"Error: Port forwarding for {service_name} failed: {error_msg}")
                        return
                else:
                    # Success! Wrapper process is running (it will auto-restart kubectl inside)
                    self.port_forward_processes.append((process, description))
                    print(f"Port forwarding started for {description} (PID: {process.pid}, auto-restart enabled)")
                    return

            except Exception as e:
                print(f"Error: Failed to start port forwarding for {service_name}: {e}")
                return

        # Timeout reached
        print(f"Error: Port forwarding for {service_name} failed to establish after {timeout}s (pod may not be ready)")

    def _stop_port_forwards(self) -> None:
        """Stop all port-forward processes and their children."""
        if not self.port_forward_processes:
            # Even if we don't have tracked processes, clean up any orphaned ones
            self._cleanup_orphaned_port_forwards()
            return

        print(f"Stopping {len(self.port_forward_processes)} port forward(s)...")

        for process, description in self.port_forward_processes:
            try:
                print(f"  Stopping {description} (PID: {process.pid})...")
                # Kill the entire process group (bash wrapper + kubectl children)
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)

                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("    Port forward didn't terminate, force killing...")
                    # Force kill if necessary
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                        process.wait(timeout=2)
                    except Exception:
                        pass
            except ProcessLookupError:
                # Process already terminated
                print(f"    Process {process.pid} already terminated")
            except Exception as e:
                print(f"    Warning: Error stopping port forward: {e}")
                # Fallback: try to kill just the parent process
                try:
                    process.kill()
                except Exception:
                    pass

        self.port_forward_processes = []

        # Final cleanup: kill any orphaned port-forward processes for this namespace
        self._cleanup_orphaned_port_forwards()

    def _cleanup_orphaned_port_forwards(self) -> None:
        """
        Clean up any orphaned port-forward processes for this namespace.

        This is a fallback cleanup that searches for and kills any remaining
        kubectl port-forward processes that are forwarding services in our namespace.
        """
        print(f"Checking for orphaned port-forward processes in namespace '{self.namespace}'...")

        try:
            # Find all port-forward processes for this namespace
            # The command line will contain: "port-forward -n <namespace>"
            search_pattern = f"port-forward.*-n {self.namespace}"

            # Use pgrep to find matching processes
            pgrep_cmd = ["pgrep", "-f", search_pattern]
            result = subprocess.run(pgrep_cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                print(f"  Found {len(pids)} orphaned port-forward process(es), cleaning up...")

                for pid in pids:
                    try:
                        pid_int = int(pid)
                        print(f"    Killing PID {pid_int}...")

                        # Use sudo if configured (same as kubectl commands)
                        if self.kubectl_sudo:
                            # Kill with sudo
                            subprocess.run(["sudo", "kill", "-TERM", str(pid_int)], stderr=subprocess.DEVNULL)
                            time.sleep(1)
                            # Force kill if still running
                            subprocess.run(["sudo", "kill", "-9", str(pid_int)], stderr=subprocess.DEVNULL)
                        else:
                            # Kill without sudo - try process group first
                            try:
                                pgid = os.getpgid(pid_int)
                                os.killpg(pgid, signal.SIGTERM)
                                time.sleep(1)
                                # Force kill if still running
                                try:
                                    os.killpg(pgid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass  # Already dead
                            except (ProcessLookupError, PermissionError):
                                # Process group kill failed, try individual process
                                try:
                                    os.kill(pid_int, signal.SIGTERM)
                                    time.sleep(1)
                                    os.kill(pid_int, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass  # Already dead
                                except PermissionError:
                                    print("      Permission denied (sudo not enabled in config)")
                    except (ValueError, Exception) as e:
                        print(f"      Warning: Could not kill PID {pid}: {e}")

                print("  Orphaned process cleanup complete")
            else:
                print("  No orphaned port-forward processes found")

        except FileNotFoundError:
            # pgrep not available, try alternative method with ps
            print("  pgrep not found, trying ps...")
            try:
                ps_cmd = ["ps", "aux"]
                result = subprocess.run(ps_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    lines = result.stdout.split("\n")
                    pids_to_kill = []

                    for line in lines:
                        if "port-forward" in line and f"-n {self.namespace}" in line:
                            # Extract PID (second column in ps aux output)
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pids_to_kill.append(int(parts[1]))
                                except ValueError:
                                    continue

                    if pids_to_kill:
                        print(f"  Found {len(pids_to_kill)} orphaned port-forward process(es), cleaning up...")
                        for pid in pids_to_kill:
                            try:
                                print(f"    Killing PID {pid}...")

                                # Use sudo if configured (same as kubectl commands)
                                if self.kubectl_sudo:
                                    # Kill with sudo
                                    subprocess.run(["sudo", "kill", "-TERM", str(pid)], stderr=subprocess.DEVNULL)
                                    time.sleep(0.5)
                                    subprocess.run(["sudo", "kill", "-9", str(pid)], stderr=subprocess.DEVNULL)
                                else:
                                    # Kill without sudo
                                    try:
                                        os.kill(pid, signal.SIGTERM)
                                        time.sleep(0.5)
                                        try:
                                            os.kill(pid, signal.SIGKILL)
                                        except ProcessLookupError:
                                            pass
                                    except PermissionError:
                                        print("      Permission denied (sudo not enabled in config)")
                                    except ProcessLookupError:
                                        pass
                            except Exception as e:
                                print(f"      Warning: Could not kill PID {pid}: {e}")
                        print("  Orphaned process cleanup complete")
                    else:
                        print("  No orphaned port-forward processes found")
            except Exception as e:
                print(f"  Warning: Could not check for orphaned processes: {e}")
        except Exception as e:
            print(f"  Warning: Error during orphaned process cleanup: {e}")

    def print_port_forward_commands(self) -> None:
        """Print commands to manually recreate port forwards."""
        port_forwards = getattr(self.config, "helm_port_forwards", None)
        if not port_forwards:
            port_forwards = [
                {
                    "service": self.release_name,
                    "local_port": 7670,
                    "remote_port": 7670,
                }
            ]

        # Group port forwards by service (same as _start_port_forwards)
        service_port_map = {}

        for pf in port_forwards:
            service_pattern = pf.get("service")
            local = pf.get("local_port")
            remote = pf.get("remote_port")

            if not all([service_pattern, local, remote]):
                continue

            # Find matching services
            service_names = self._find_services_by_pattern(service_pattern)

            # Add ports to each matching service
            for service_name in service_names:
                if service_name not in service_port_map:
                    service_port_map[service_name] = []
                service_port_map[service_name].append((local, remote))

        print("\nTo manually recreate port forwards (with auto-restart), run:")
        print("=" * 60)
        for service_name, port_pairs in service_port_map.items():
            port_strs = [f"{local}:{remote}" for local, remote in port_pairs]
            kubectl_cmd_str = " ".join(self.kubectl_cmd)
            ports_arg = " ".join(port_strs)
            # Show resilient version with auto-restart loop
            print(f"  # Auto-restarting port-forward for {service_name}:")
            print(
                (
                    f"  while true; do {kubectl_cmd_str} port-forward -n {self.namespace} "
                    f"service/{service_name} {ports_arg}; sleep 5; done &"
                )
            )
            print()
        print("=" * 60)

    def stop(self, clean: bool = False) -> int:
        """
        Uninstall Helm release.

        Note: Port forwards should be stopped separately via _stop_port_forwards()
        to allow keeping services up while cleaning up port-forward processes.

        Args:
            clean: If True, also delete the namespace (full cleanup)

        Returns:
            0 on success, non-zero on failure
        """
        print(f"Uninstalling Helm release {self.release_name}...")

        # Stop port forwards first
        self._stop_port_forwards()

        cmd = self.helm_cmd + ["uninstall", self.release_name, "--namespace", self.namespace]

        rc = run_cmd(cmd)
        if rc != 0:
            print(f"Warning: helm uninstall returned {rc}")

        # If clean mode, also delete the namespace for a complete cleanup
        if clean:
            print(f"Deleting namespace {self.namespace}...")
            kubectl_cmd = self.kubectl_cmd + ["delete", "namespace", self.namespace, "--ignore-not-found"]
            rc2 = run_cmd(kubectl_cmd)
            if rc2 != 0:
                print(f"Warning: kubectl delete namespace returned {rc2}")

        return 0

    def check_readiness(self, timeout_s: int, check_milvus: bool = True) -> bool:
        """
        Check readiness by polling HTTP endpoint.

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
        Get service URL for Helm deployment.

        Args:
            service: Service name ("api" or "health")

        Returns:
            URL string
        """
        # Use hostname from config
        # For Helm deployments, this is the port-forwarded localhost
        hostname = getattr(self.config, "hostname", "localhost")

        if service == "health":
            return f"http://{hostname}:7670/v1/health/ready"
        return f"http://{hostname}:7670"
