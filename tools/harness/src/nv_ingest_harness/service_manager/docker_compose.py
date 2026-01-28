"""Docker Compose service manager implementation."""

import subprocess
import time
import urllib.request
from pathlib import Path

from nv_ingest_harness.service_manager.base import ServiceManager
from nv_ingest_harness.utils.interact import run_cmd


class DockerComposeManager(ServiceManager):
    """Manages services using Docker Compose."""

    def __init__(self, config, repo_root: Path, sku: str | None = None):
        """
        Initialize Docker Compose manager.

        Args:
            config: Configuration object with profiles and settings
            repo_root: Path to the repository root
            sku: Optional GPU SKU for override file (e.g., a10g, a100-40gb, l40s)
        """
        super().__init__(config, repo_root)
        self.compose_file = str(repo_root / "docker-compose.yaml")
        self.sku = sku
        self.override_file = None

        # Set override file path if sku is provided
        if self.sku:
            override_path = repo_root / f"docker-compose.{self.sku}.yaml"
            if override_path.exists():
                self.override_file = str(override_path)
                print(f"Using Docker Compose override file: {self.override_file}")
            else:
                print(f"Warning: Override file not found: {override_path}")

    def _build_compose_cmd(self, base_cmd: list[str]) -> list[str]:
        """
        Build docker compose command with compose file and optional override file.

        Args:
            base_cmd: Base command to start with (e.g., ["docker", "compose"])

        Returns:
            Command list with -f flags for compose file(s)
        """
        cmd = base_cmd + ["-f", self.compose_file]
        if self.override_file:
            cmd += ["-f", self.override_file]
        return cmd

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
        cmd = self._build_compose_cmd(["docker", "compose"])
        cmd += ["--profile", profile_list[0]]
        for p in profile_list[1:]:
            cmd += ["--profile", p]

        # Add up command with or without build
        if not no_build:
            cmd += ["up", "--build", "-d"]
        else:
            cmd += ["up", "-d"]

        return run_cmd(cmd)

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
        down_cmd = self._build_compose_cmd(["docker", "compose"])
        down_cmd += ["--profile", "*", "down"]

        # Add cleanup flags if clean mode
        if clean:
            down_cmd += ["-v", "--remove-orphans"]

        rc = run_cmd(down_cmd)
        if rc != 0:
            print(f"Warning: docker compose down returned {rc}")

        # Remove containers forcefully
        rm_cmd = self._build_compose_cmd(["docker", "compose"])
        rm_cmd += ["--profile", "*", "rm", "--force"]
        rc2 = run_cmd(rm_cmd)
        if rc2 != 0:
            print(f"Warning: docker compose rm returned {rc2}")

        return 0

    def check_readiness(self, timeout_s: int, check_milvus: bool = True, check_embedding: bool = True) -> bool:
        """
        Poll the health endpoint until ready.

        Args:
            timeout_s: Timeout in seconds
            check_milvus: If True, also check Milvus health endpoint
            check_embedding: If True, also check embedding service health endpoint

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
                        all_services_ready = True

                        # If Milvus check is enabled, verify it's also ready
                        if check_milvus:
                            hostname = getattr(self.config, "hostname", "localhost")
                            milvus_url = f"http://{hostname}:9091/healthz"
                            try:
                                with urllib.request.urlopen(milvus_url, timeout=5) as milvus_resp:
                                    if milvus_resp.status != 200:
                                        all_services_ready = False
                            except Exception:
                                all_services_ready = False

                        # If embedding check is enabled, verify it's also ready
                        if check_embedding:
                            hostname = getattr(self.config, "hostname", "localhost")
                            embedding_url = f"http://{hostname}:8012/v1/health/ready"
                            try:
                                with urllib.request.urlopen(embedding_url, timeout=5) as embedding_resp:
                                    if embedding_resp.status != 200:
                                        all_services_ready = False
                            except Exception:
                                all_services_ready = False

                        if all_services_ready:
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

    def dump_logs(self, artifacts_dir: Path) -> int:
        """
        Dump logs of all Docker Compose containers to artifacts directory.

        Args:
            artifacts_dir: Directory to write log files to

        Returns:
            0 on success, non-zero on failure
        """
        print(f"Dumping Docker Compose logs to {artifacts_dir}...")

        # Ensure artifacts directory exists
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Get list of running containers from this compose project
        ps_cmd = self._build_compose_cmd(["docker", "compose"])
        ps_cmd += ["--profile", "*", "ps", "-q"]
        try:
            result = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"Warning: Could not list containers: {result.stderr.strip()}")
                return result.returncode

            container_ids = [cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()]

            if not container_ids:
                print("No containers found to dump logs from")
                return 0

            print(f"Found {len(container_ids)} container(s) to dump logs from")

            # Dump logs for each container individually with container name
            for container_id in container_ids:
                # Get container name
                inspect_cmd = ["docker", "inspect", "--format", "{{.Name}}", container_id]
                name_result = subprocess.run(inspect_cmd, capture_output=True, text=True, timeout=10)
                if name_result.returncode == 0:
                    container_name = name_result.stdout.strip().lstrip("/")
                else:
                    container_name = container_id[:12]  # Use short ID as fallback

                log_file = artifacts_dir / f"container_{container_name}.log"
                print(f"  Dumping logs for {container_name} to {log_file.name}")

                # Dump container logs
                logs_cmd = ["docker", "logs", container_id]
                with open(log_file, "w") as f:
                    log_result = subprocess.run(logs_cmd, stdout=f, stderr=subprocess.STDOUT, timeout=60)
                    if log_result.returncode != 0:
                        print(f"    Warning: Failed to dump logs for {container_name}")

            # Also dump combined logs from all services
            combined_log_file = artifacts_dir / "docker_compose_combined.log"
            print(f"  Dumping combined logs to {combined_log_file.name}")
            logs_cmd = self._build_compose_cmd(["docker", "compose"])
            logs_cmd += ["--profile", "*", "logs", "--no-color"]
            with open(combined_log_file, "w") as f:
                combined_result = subprocess.run(logs_cmd, stdout=f, stderr=subprocess.STDOUT, timeout=120)
                if combined_result.returncode != 0:
                    print("    Warning: Failed to dump combined logs")

            print("Log dump complete")
            return 0

        except subprocess.TimeoutExpired:
            print("Error: Timeout while dumping logs")
            return 1
        except Exception as e:
            print(f"Error: Failed to dump logs: {e}")
            return 1
