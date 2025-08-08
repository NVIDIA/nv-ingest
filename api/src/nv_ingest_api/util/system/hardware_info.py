import logging
import os
import platform
from typing import Optional, Dict, Any, Tuple

# Try importing psutil, but don't make it a hard requirement if only cgroups are needed
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)

# --- Cgroup Constants ---
CGROUP_V1_CPU_DIR = "/sys/fs/cgroup/cpu"
CGROUP_V1_CPUACCT_DIR = "/sys/fs/cgroup/cpuacct"  # Sometimes usage is here
CGROUP_V2_CPU_FILE = "/sys/fs/cgroup/cpu.max"  # Standard path in v2 unified hierarchy

# Memory cgroup paths
CGROUP_V1_MEMORY_DIR = "/sys/fs/cgroup/memory"
CGROUP_V2_MEMORY_FILE = "/sys/fs/cgroup/memory.max"  # v2 unified hierarchy
CGROUP_V2_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"  # Current usage in v2


class SystemResourceProbe:
    """
    Detects the effective CPU core count available to the current process,
    optionally applying a weighting factor for hyperthreads (SMT).

    It attempts to reconcile information from:
    1. Linux Cgroup v2 CPU limits (cpu.max)
    2. Linux Cgroup v1 CPU limits (cpu.cfs_quota_us, cpu.cfs_period_us)
    3. OS scheduler affinity (os.sched_getaffinity)
    4. OS reported CPU counts (psutil.cpu_count for logical/physical)

    Prioritizes Cgroup quota limits. If the limit is based on core count
    (affinity/OS), it applies hyperthreading weight if psutil provides
    physical/logical counts.
    """

    def __init__(self, hyperthread_weight: float = 0.75):
        """
        Initializes the detector and performs the detection.

        Parameters
        ----------
        hyperthread_weight : float, optional
            The performance weighting factor for hyperthreads (0.0 to 1.0).
            A value of 1.0 treats hyperthreads the same as physical cores.
            A value of 0.5 suggests a hyperthread adds 50% extra performance.
            Requires psutil to be installed and report physical cores.
            Defaults to 0.75.

            Note: the default value of 0.75 is a heuristic and may not be optimal
            for all situations. It is where parallel pdf decomposition efficiency
            is observed to begin rolling off.
        """
        if not (0.0 <= hyperthread_weight <= 1.0):
            raise ValueError("hyperthread_weight must be between 0.0 and 1.0")

        self.hyperthread_weight: float = hyperthread_weight if psutil else 1.0  # Force 1.0 if psutil missing
        if not psutil and hyperthread_weight != 1.0:
            logger.warning("psutil not found. Hyperthreading weight ignored (effectively 1.0).")

        # OS Info
        self.os_logical_cores: Optional[int] = None
        self.os_physical_cores: Optional[int] = None
        self.os_sched_affinity_cores: Optional[int] = None

        # Cgroup Info
        self.cgroup_type: Optional[str] = None
        self.cgroup_quota_cores: Optional[float] = None
        self.cgroup_period_us: Optional[int] = None
        self.cgroup_shares: Optional[int] = None
        self.cgroup_usage_percpu_us: Optional[list[int]] = None
        self.cgroup_usage_total_us: Optional[int] = None

        # Memory Info
        self.os_total_memory_bytes: Optional[int] = None
        self.cgroup_memory_limit_bytes: Optional[int] = None
        self.cgroup_memory_usage_bytes: Optional[int] = None
        self.effective_memory_bytes: Optional[int] = None
        self.memory_detection_method: str = "unknown"

        # --- Result ---
        # Raw limit before potential weighting
        self.raw_limit_value: Optional[float] = None
        self.raw_limit_method: str = "unknown"
        # Final potentially weighted result
        self.effective_cores: Optional[float] = None
        self.detection_method: str = "unknown"  # Method for the final effective_cores

        self._detect()

    @staticmethod
    def _read_file_int(path: str) -> Optional[int]:
        """Safely reads an integer from a file."""
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    content = f.read().strip()
                    if content:
                        return int(content)
        except (IOError, ValueError, PermissionError) as e:
            logger.debug(f"Failed to read or parse int from {path}: {e}")
        return None

    @staticmethod
    def _read_file_str(path: str) -> Optional[str]:
        """Safely reads a string from a file."""
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read().strip()
        except (IOError, PermissionError) as e:
            logger.debug(f"Failed to read string from {path}: {e}")
        return None

    def _read_cgroup_v1(self) -> bool:
        """Attempts to read Cgroup v1 CPU limits."""
        if not os.path.exists(CGROUP_V1_CPU_DIR):
            logger.debug(f"Cgroup v1 CPU dir not found: {CGROUP_V1_CPU_DIR}")
            return False

        logger.debug(f"Checking Cgroup v1 limits in {CGROUP_V1_CPU_DIR}")
        quota_us = self._read_file_int(os.path.join(CGROUP_V1_CPU_DIR, "cpu.cfs_quota_us"))
        period_us = self._read_file_int(os.path.join(CGROUP_V1_CPU_DIR, "cpu.cfs_period_us"))
        shares = self._read_file_int(os.path.join(CGROUP_V1_CPU_DIR, "cpu.shares"))

        # Check cpuacct for usage stats if dir exists
        if os.path.exists(CGROUP_V1_CPUACCT_DIR):
            usage_total = self._read_file_int(os.path.join(CGROUP_V1_CPUACCT_DIR, "cpuacct.usage"))
            usage_percpu_str = self._read_file_str(os.path.join(CGROUP_V1_CPUACCT_DIR, "cpuacct.usage_percpu"))
            if usage_percpu_str:
                try:
                    self.cgroup_usage_percpu_us = [int(x) for x in usage_percpu_str.split()]
                except ValueError:
                    logger.warning("Could not parse cpuacct.usage_percpu")
            if usage_total is not None:
                self.cgroup_usage_total_us = usage_total

        if quota_us is not None and period_us is not None:
            self.cgroup_type = "v1"
            self.cgroup_period_us = period_us
            self.cgroup_shares = shares  # May be None if file doesn't exist/readable

            if quota_us > 0 and period_us > 0:
                self.cgroup_quota_cores = quota_us / period_us
                logger.debug(
                    f"Cgroup v1 quota detected: {quota_us} us / {period_us} us = {self.cgroup_quota_cores:.2f}"
                    f" effective cores"
                )
                return True
            elif quota_us == -1:
                logger.debug("Cgroup v1 quota detected: Unlimited (-1)")
                # No quota limit, but we know it's cgroup v1
                return True  # Return true because we identified the type
            else:
                logger.warning(f"Cgroup v1 quota/period values invalid? Quota: {quota_us}, Period: {period_us}")

        elif shares is not None:  # If only shares are readable, still note it's v1
            self.cgroup_type = "v1"
            self.cgroup_shares = shares
            logger.debug(f"Cgroup v1 shares detected: {shares} (no quota found)")
            return True

        return False

    def _read_cgroup_v2(self) -> bool:
        """Attempts to read Cgroup v2 CPU limits."""
        if not os.path.exists(CGROUP_V2_CPU_FILE):
            logger.debug(f"Cgroup v2 cpu.max file not found: {CGROUP_V2_CPU_FILE}")
            return False

        logger.debug(f"Checking Cgroup v2 limits in {CGROUP_V2_CPU_FILE}")
        content = self._read_file_str(CGROUP_V2_CPU_FILE)
        if content:
            self.cgroup_type = "v2"
            parts = content.split()
            if len(parts) == 2:
                quota_str, period_str = parts
                try:
                    period_us = int(period_str)
                    self.cgroup_period_us = period_us
                    if quota_str == "max":
                        logger.debug("Cgroup v2 quota detected: Unlimited ('max')")
                        return True  # Identified type, no quota limit
                    else:
                        quota_us = int(quota_str)
                        if quota_us > 0 and period_us > 0:
                            self.cgroup_quota_cores = quota_us / period_us
                            logger.debug(
                                f"Cgroup v2 quota detected: {quota_us} us / {period_us}"
                                f" us = {self.cgroup_quota_cores:.2f} effective cores"
                            )
                            return True
                        else:
                            logger.warning(
                                f"Cgroup v2 quota/period values invalid? Quota: {quota_us}, Period: {period_us}"
                            )

                except ValueError:
                    logger.warning(f"Could not parse Cgroup v2 cpu.max content: '{content}'")
            else:
                logger.warning(f"Unexpected format in Cgroup v2 cpu.max: '{content}'")
        return False

    @staticmethod
    def _get_os_affinity() -> Optional[int]:
        """Gets CPU count via os.sched_getaffinity."""
        if platform.system() != "Linux":
            logger.debug("os.sched_getaffinity is Linux-specific.")
            return None
        try:
            # sched_getaffinity exists on Linux
            affinity = os.sched_getaffinity(0)  # 0 for current process
            count = len(affinity)
            if count > 0:
                logger.debug(f"Detected {count} cores via os.sched_getaffinity.")
                return count
            else:
                logger.warning("os.sched_getaffinity(0) returned 0 or empty set.")
                return None
        except AttributeError:
            logger.debug("os.sched_getaffinity not available on this platform/Python version.")
            return None
        except OSError as e:
            logger.warning(f"Could not get affinity: {e}")
            return None

    @staticmethod
    def _get_os_cpu_counts() -> Tuple[Optional[int], Optional[int]]:
        """Gets logical and physical CPU counts using psutil or os.cpu_count."""
        logical = None
        physical = None
        source = "unknown"

        if psutil:
            try:
                logical = psutil.cpu_count(logical=True)
                physical = psutil.cpu_count(logical=False)
                source = "psutil"
                if not logical:
                    logical = None  # Ensure None if psutil returns 0/None
                if not physical:
                    physical = None
            except Exception as e:
                logger.warning(f"psutil.cpu_count failed: {e}. Falling back to os.cpu_count.")
                logical, physical = None, None  # Reset before fallback

        if logical is None:  # Fallback if psutil failed or not installed
            try:
                logical = os.cpu_count()
                source = "os.cpu_count"
                # os.cpu_count doesn't usually provide physical count, leave as None
            except NotImplementedError:
                logger.error("os.cpu_count() is not implemented on this system.")
            except Exception as e:
                logger.error(f"os.cpu_count() failed: {e}")

        if logical:
            logger.debug(f"Detected {logical} logical cores via {source}.")
        if physical:
            logger.debug(f"Detected {physical} physical cores via {source}.")

        return logical, physical

    # --- Weighting Function ---
    def _apply_hyperthread_weight(self, logical_limit: int) -> float:
        """
        Applies hyperthreading weight to an integer logical core limit.

        Parameters
        ----------
        logical_limit : int
            The maximum number of logical cores allowed (e.g., from affinity or OS count).

        Returns
        -------
        float
            The estimated effective core performance based on weighting.
            Returns logical_limit if weighting cannot be applied.
        """
        P = self.os_physical_cores
        # Weighting requires knowing both physical and logical counts
        if P is not None and P > 0 and self.os_logical_cores is not None:
            # Apply the heuristic: P physical cores + (N-P) hyperthreads * weight
            # Ensure N is capped by the actual number of logical cores available
            N = min(logical_limit, self.os_logical_cores)

            physical_part = min(N, P)
            hyperthread_part = max(0, N - P)

            weighted_cores = (physical_part * 1.0) + (hyperthread_part * self.hyperthread_weight)

            if weighted_cores != N:  # Log only if weighting changes the value
                logger.debug(
                    f"Applying hyperthread weight ({self.hyperthread_weight:.2f}) to "
                    f"logical limit {logical_limit} (System: {P}P/{self.os_logical_cores}L): "
                    f"Effective weighted cores = {weighted_cores:.2f}"
                )
            else:
                logger.debug(
                    f"Hyperthread weighting ({self.hyperthread_weight:.2f}) applied to "
                    f"logical limit {logical_limit} (System: {P}P/{self.os_logical_cores}L), "
                    f"but result is still {weighted_cores:.2f} (e.g., limit <= physical or weight=1.0)"
                )
            return weighted_cores
        else:
            # Cannot apply weighting
            if self.hyperthread_weight != 1.0:  # Only warn if weighting was requested
                if not psutil:
                    # Already warned about missing psutil during init
                    pass
                elif P is None:
                    logger.warning("Cannot apply hyperthread weight: Physical core count not available.")
                else:  # L must be missing
                    logger.warning("Cannot apply hyperthread weight: Logical core count not available.")

            logger.debug(f"Skipping hyperthread weight calculation for logical limit {logical_limit}.")
            return float(logical_limit)  # Return the original limit as float

    # --- Memory Detection Methods ---
    @staticmethod
    def _get_os_memory() -> Optional[int]:
        """Gets total system memory in bytes using psutil or /proc/meminfo."""
        # Try psutil first
        if psutil:
            try:
                memory = psutil.virtual_memory()
                total_bytes = memory.total
                if total_bytes and total_bytes > 0:
                    logger.debug(f"Detected {total_bytes / (1024**3):.2f} GB system memory via psutil.")
                    return total_bytes
            except Exception as e:
                logger.warning(f"psutil.virtual_memory() failed: {e}. Falling back to /proc/meminfo.")

        # Fallback to /proc/meminfo
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # MemTotal is in KB
                            parts = line.split()
                            if len(parts) >= 2:
                                total_kb = int(parts[1])
                                total_bytes = total_kb * 1024
                                logger.debug(
                                    f"Detected {total_bytes / (1024**3):.2f} GB system memory via /proc/meminfo."
                                )
                                return total_bytes
                            break
        except (IOError, ValueError, PermissionError) as e:
            logger.warning(f"Failed to read /proc/meminfo: {e}")

        logger.error("Could not determine system memory from any source.")
        return None

    def _read_memory_cgroup_v2(self) -> bool:
        """Attempts to read Cgroup v2 memory limits."""
        if not os.path.exists(CGROUP_V2_MEMORY_FILE):
            logger.debug(f"Cgroup v2 memory.max file not found: {CGROUP_V2_MEMORY_FILE}")
            return False

        logger.debug(f"Checking Cgroup v2 memory limits in {CGROUP_V2_MEMORY_FILE}")
        content = self._read_file_str(CGROUP_V2_MEMORY_FILE)
        if content:
            try:
                if content == "max":
                    logger.debug("Cgroup v2 memory limit: unlimited")
                    return True
                else:
                    limit_bytes = int(content)
                    self.cgroup_memory_limit_bytes = limit_bytes
                    logger.debug(f"Cgroup v2 memory limit: {limit_bytes / (1024**3):.2f} GB")

                    # Also try to read current usage
                    usage_content = self._read_file_str(CGROUP_V2_MEMORY_CURRENT)
                    if usage_content:
                        try:
                            usage_bytes = int(usage_content)
                            self.cgroup_memory_usage_bytes = usage_bytes
                            logger.debug(f"Cgroup v2 memory usage: {usage_bytes / (1024**3):.2f} GB")
                        except ValueError:
                            logger.debug(f"Could not parse memory.current: '{usage_content}'")

                    return True
            except ValueError:
                logger.warning(f"Could not parse Cgroup v2 memory.max content: '{content}'")
        return False

    def _read_memory_cgroup_v1(self) -> bool:
        """Attempts to read Cgroup v1 memory limits."""
        if not os.path.exists(CGROUP_V1_MEMORY_DIR):
            logger.debug(f"Cgroup v1 memory dir not found: {CGROUP_V1_MEMORY_DIR}")
            return False

        logger.debug(f"Checking Cgroup v1 memory limits in {CGROUP_V1_MEMORY_DIR}")

        # Try memory.limit_in_bytes
        limit_bytes = self._read_file_int(os.path.join(CGROUP_V1_MEMORY_DIR, "memory.limit_in_bytes"))
        usage_bytes = self._read_file_int(os.path.join(CGROUP_V1_MEMORY_DIR, "memory.usage_in_bytes"))

        if limit_bytes is not None:
            # Cgroup v1 often shows very large values (like 9223372036854775807) for unlimited
            # We consider values >= 2^63-1 or >= system memory * 100 as unlimited
            if limit_bytes >= 9223372036854775807 or (
                self.os_total_memory_bytes and limit_bytes >= self.os_total_memory_bytes * 100
            ):
                logger.debug("Cgroup v1 memory limit: unlimited (very large value)")
                return True
            else:
                self.cgroup_memory_limit_bytes = limit_bytes
                logger.debug(f"Cgroup v1 memory limit: {limit_bytes / (1024**3):.2f} GB")

                if usage_bytes is not None:
                    self.cgroup_memory_usage_bytes = usage_bytes
                    logger.debug(f"Cgroup v1 memory usage: {usage_bytes / (1024**3):.2f} GB")

                return True

        return False

    def _detect_memory(self):
        """Performs memory detection sequence."""
        logger.debug("Starting memory detection...")

        # 1. Get OS level memory first
        self.os_total_memory_bytes = self._get_os_memory()

        # 2. Try Cgroup v2 memory limits
        cgroup_memory_detected = self._read_memory_cgroup_v2()

        # 3. Try Cgroup v1 if v2 not found or didn't yield a limit
        if not cgroup_memory_detected or self.cgroup_memory_limit_bytes is None:
            cgroup_memory_detected = self._read_memory_cgroup_v1()

        # 4. Determine effective memory
        if self.cgroup_memory_limit_bytes is not None and self.os_total_memory_bytes is not None:
            # Use the smaller of cgroup limit and system memory
            self.effective_memory_bytes = min(self.cgroup_memory_limit_bytes, self.os_total_memory_bytes)
            self.memory_detection_method = "cgroup_limited"
            logger.debug(f"Effective memory: {self.effective_memory_bytes / (1024**3):.2f} GB (cgroup limited)")
        elif self.os_total_memory_bytes is not None:
            # No cgroup limit, use system memory
            self.effective_memory_bytes = self.os_total_memory_bytes
            self.memory_detection_method = "system_memory"
            logger.debug(f"Effective memory: {self.effective_memory_bytes / (1024**3):.2f} GB (system memory)")
        else:
            logger.error("Could not determine effective memory limit")
            self.memory_detection_method = "failed"

    def _detect(self):
        """Performs the detection sequence and applies weighting."""
        logger.debug("Starting effective core count detection...")

        # 1. Get OS level counts first
        self.os_logical_cores, self.os_physical_cores = self._get_os_cpu_counts()

        # 2. Try Cgroup v2
        cgroup_detected = self._read_cgroup_v2()

        # 3. Try Cgroup v1 if v2 not found or didn't yield quota
        if not cgroup_detected or (self.cgroup_type == "v2" and self.cgroup_quota_cores is None):
            cgroup_detected = self._read_cgroup_v1()

        # 4. Get OS Affinity
        self.os_sched_affinity_cores = self._get_os_affinity()

        # 5. Detect Memory
        self._detect_memory()

        # --- 6. Determine the RAW Limit (before weighting) ---
        raw_limit = float("inf")
        raw_method = "unknown"

        # Priority 1: Cgroup Quota
        if self.cgroup_quota_cores is not None and self.cgroup_quota_cores > 0:
            raw_limit = min(raw_limit, self.cgroup_quota_cores)
            raw_method = f"cgroup_{self.cgroup_type}_quota"
            logger.debug(f"Raw limit set by Cgroup Quota: {self.cgroup_quota_cores:.2f}")

        # Priority 2: Scheduler Affinity
        if self.os_sched_affinity_cores is not None and self.os_sched_affinity_cores > 0:
            affinity_limit = float(self.os_sched_affinity_cores)
            if affinity_limit < raw_limit:
                raw_limit = affinity_limit
                raw_method = "sched_affinity"
                logger.debug(f"Raw limit updated by Sched Affinity: {affinity_limit}")
            elif raw_method.startswith("cgroup"):
                logger.debug(
                    f"Sched Affinity limit ({affinity_limit}) not stricter than Cgroup Quota ({raw_limit:.2f})."
                )

        # Priority 3: OS Logical Cores
        if raw_limit == float("inf"):  # If no cgroup quota or affinity was found/applied
            if self.os_logical_cores is not None and self.os_logical_cores > 0:
                raw_limit = float(self.os_logical_cores)
                raw_method = "os_logical_count"
                logger.debug(f"Raw limit set by OS Logical Core count: {self.os_logical_cores}")
            else:
                # Absolute fallback
                logger.warning("Could not determine any CPU core limit. Defaulting raw limit to 1.0.")
                raw_limit = 1.0
                raw_method = "fallback_default"

        self.raw_limit_value = raw_limit
        self.raw_limit_method = raw_method
        logger.debug(f"Raw CPU limit determined: {self.raw_limit_value:.2f} (Method: {self.raw_limit_method})")

        # --- 7. Apply Weighting (if applicable) ---
        final_effective_cores = raw_limit
        final_method = raw_method

        # Apply weighting ONLY if the raw limit is NOT from a cgroup quota
        # AND the limit is an integer (or effectively integer) core count
        if not raw_method.startswith("cgroup_"):
            # Check if raw_limit is effectively an integer
            if abs(raw_limit - round(raw_limit)) < 1e-9 and raw_limit > 0:
                logical_limit_int = int(round(raw_limit))
                weighted_value = self._apply_hyperthread_weight(logical_limit_int)
                final_effective_cores = weighted_value
                # Update method if weighting was actually applied and changed the value
                if abs(weighted_value - raw_limit) > 1e-9:
                    final_method = f"{raw_method}_weighted"
                else:
                    # Keep original method name if weighting didn't change result
                    final_method = raw_method

            else:  # Raw limit was affinity/os count but not an integer? Should be rare.
                logger.debug(
                    f"Raw limit method '{raw_method}' is not cgroup quota, "
                    f"but value {raw_limit:.2f} is not integer. Skipping weighting."
                )

        elif raw_method.startswith("cgroup_"):
            logger.debug("Raw limit is from Cgroup quota. Using quota value directly (skipping SMT weighting).")

        self.effective_cores = final_effective_cores
        self.detection_method = final_method  # The method for the final value

        logger.debug(
            f"Effective CPU core limit determined: {self.effective_cores:.2f} " f"(Method: {self.detection_method})"
        )

    def get_effective_cores(self) -> Optional[float]:
        """Returns the primary result: the effective core limit, potentially weighted."""
        return self.effective_cores

    @property
    def total_memory_mb(self) -> Optional[float]:
        """Returns the effective memory limit in megabytes."""
        if self.effective_memory_bytes is not None:
            return self.effective_memory_bytes / (1024 * 1024)
        return None

    @property
    def cpu_count(self) -> Optional[float]:
        """Returns the effective CPU count for compatibility."""
        return self.effective_cores

    def get_details(self) -> Dict[str, Any]:
        """Returns a dictionary with all detected information."""
        # Calculate full system weighted potential for info
        os_weighted_cores = None
        if self.os_physical_cores and self.os_logical_cores:
            # Use weighting func with the total logical cores as the limit
            os_weighted_cores = self._apply_hyperthread_weight(self.os_logical_cores)

        return {
            "effective_cores": self.effective_cores,
            "detection_method": self.detection_method,
            "raw_limit_value": self.raw_limit_value,
            "raw_limit_method": self.raw_limit_method,
            "hyperthread_weight_applied": self.hyperthread_weight,
            "os_logical_cores": self.os_logical_cores,
            "os_physical_cores": self.os_physical_cores,
            "os_weighted_potential": os_weighted_cores,  # Full system potential weighted
            "os_sched_affinity_cores": self.os_sched_affinity_cores,
            "cgroup_type": self.cgroup_type,
            "cgroup_quota_cores": self.cgroup_quota_cores,
            "cgroup_period_us": self.cgroup_period_us,
            "cgroup_shares": self.cgroup_shares,
            "cgroup_usage_total_us": self.cgroup_usage_total_us,
            "cgroup_usage_percpu_us": self.cgroup_usage_percpu_us,
            # Memory information
            "effective_memory_bytes": self.effective_memory_bytes,
            "effective_memory_mb": self.total_memory_mb,
            "memory_detection_method": self.memory_detection_method,
            "os_total_memory_bytes": self.os_total_memory_bytes,
            "cgroup_memory_limit_bytes": self.cgroup_memory_limit_bytes,
            "cgroup_memory_usage_bytes": self.cgroup_memory_usage_bytes,
            "platform": platform.system(),
        }
