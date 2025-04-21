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


class CoreCountDetector:
    """
    Detects the effective CPU core count available to the current process.

    It attempts to reconcile information from:
    1. Linux Cgroup v2 CPU limits (cpu.max)
    2. Linux Cgroup v1 CPU limits (cpu.cfs_quota_us, cpu.cfs_period_us, cpu.shares)
    3. OS scheduler affinity (os.sched_getaffinity)
    4. OS reported CPU counts (os.cpu_count / psutil.cpu_count)

    Prioritizes Cgroup quota limits if set, as they represent hard time-slice
    restrictions often used in container orchestrators like Kubernetes/Helm.
    Falls back to affinity, then OS count.
    """

    def __init__(self):
        """Initializes the detector and performs the detection."""
        self.os_logical_cores: Optional[int] = None
        self.os_physical_cores: Optional[int] = None
        self.os_sched_affinity_cores: Optional[int] = None
        self.cgroup_type: Optional[str] = None  # 'v1' or 'v2'
        self.cgroup_quota_cores: Optional[float] = None
        self.cgroup_period_us: Optional[int] = None
        self.cgroup_shares: Optional[int] = None  # v1 only
        self.cgroup_usage_percpu_us: Optional[list[int]] = None  # v1 cpuacct
        self.cgroup_usage_total_us: Optional[int] = None  # v1 cpuacct

        self.detection_method: str = "unknown"
        self.effective_cores: Optional[float] = None  # Can be fractional due to quota

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
                logger.info(
                    f"Cgroup v1 quota detected: {quota_us} us / {period_us} us = {self.cgroup_quota_cores:.2f}"
                    f" effective cores"
                )
                return True
            elif quota_us == -1:
                logger.info("Cgroup v1 quota detected: Unlimited (-1)")
                # No quota limit, but we know it's cgroup v1
                return True  # Return true because we identified the type
            else:
                logger.warning(f"Cgroup v1 quota/period values invalid? Quota: {quota_us}, Period: {period_us}")

        elif shares is not None:  # If only shares are readable, still note it's v1
            self.cgroup_type = "v1"
            self.cgroup_shares = shares
            logger.info(f"Cgroup v1 shares detected: {shares} (no quota found)")
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
                        logger.info("Cgroup v2 quota detected: Unlimited ('max')")
                        return True  # Identified type, no quota limit
                    else:
                        quota_us = int(quota_str)
                        if quota_us > 0 and period_us > 0:
                            self.cgroup_quota_cores = quota_us / period_us
                            logger.info(
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
                logger.info(f"Detected {count} cores via os.sched_getaffinity.")
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
            logger.info(f"Detected {logical} logical cores via {source}.")
        if physical:
            logger.info(f"Detected {physical} physical cores via {source}.")

        return logical, physical

    def _detect(self):
        """Performs the detection sequence."""
        logger.debug("Starting effective core count detection...")

        # 1. Get OS level counts first for context
        self.os_logical_cores, self.os_physical_cores = self._get_os_cpu_counts()

        # 2. Try Cgroup v2 (preferred modern standard)
        cgroup_detected = self._read_cgroup_v2()

        # 3. Try Cgroup v1 if v2 not found or didn't yield quota
        if not cgroup_detected or (self.cgroup_type == "v2" and self.cgroup_quota_cores is None):
            if not cgroup_detected:  # Only log if we haven't already found v2
                logger.debug("Cgroup v2 not detected or no quota found, trying v1.")
            cgroup_detected = self._read_cgroup_v1()

        # 4. Get OS Affinity
        self.os_sched_affinity_cores = self._get_os_affinity()

        # 5. Determine Effective Cores based on priority: Cgroup Quota > Affinity > OS Logical
        final_limit = float("inf")
        method = "os_logical_count"  # Default if nothing else found

        # Priority 1: Cgroup Quota (if defined and positive)
        if self.cgroup_quota_cores is not None and self.cgroup_quota_cores > 0:
            final_limit = min(final_limit, self.cgroup_quota_cores)
            method = f"cgroup_{self.cgroup_type}_quota"
            logger.debug(f"Applying Cgroup Quota limit: {self.cgroup_quota_cores:.2f}")

        # Priority 2: Scheduler Affinity (if defined and positive)
        if self.os_sched_affinity_cores is not None and self.os_sched_affinity_cores > 0:
            # If we already have a cgroup quota, the effective limit is the MINIMUM
            # of the two. You can't use more cores than affinity allows, AND you
            # can't use more CPU time than the quota allows.
            if method.startswith("cgroup"):
                if self.os_sched_affinity_cores < final_limit:
                    logger.info(
                        f"Refining Cgroup quota limit ({final_limit:.2f}) with smaller sched_affinity limit"
                        f" ({self.os_sched_affinity_cores})"
                    )
                    final_limit = float(self.os_sched_affinity_cores)
                    method = "sched_affinity_capped_by_cgroup"  # Or just sched_affinity? Let's be clear.
                else:
                    # CGroup limit is already stricter or equal to affinity limit
                    logger.debug(
                        f"Sched_affinity limit ({self.os_sched_affinity_cores}) is not stricter than Cgroup quota"
                        f" ({final_limit:.2f}). Keeping Cgroup limit."
                    )
            else:
                # No Cgroup quota, affinity is the primary limit found so far
                final_limit = min(final_limit, float(self.os_sched_affinity_cores))
                method = "sched_affinity"
                logger.debug(f"Applying Sched Affinity limit: {self.os_sched_affinity_cores}")

        # Priority 3: OS Logical Cores (as fallback)
        if final_limit == float("inf"):  # If no cgroup quota or affinity was found/applied
            if self.os_logical_cores is not None and self.os_logical_cores > 0:
                final_limit = float(self.os_logical_cores)
                method = "os_logical_count"
                logger.debug(f"Applying OS Logical Core count limit: {self.os_logical_cores}")
            else:
                # Absolute fallback - should be rare
                logger.warning("Could not determine any CPU core limit. Defaulting to 1.")
                final_limit = 1.0
                method = "fallback_default"

        self.effective_cores = final_limit
        self.detection_method = method
        logger.info(
            f"Effective CPU core limit determined: {self.effective_cores:.2f} (Method: {self.detection_method})"
        )

    def get_effective_cores(self) -> Optional[float]:
        """Returns the primary result: the effective core limit."""
        return self.effective_cores

    def get_details(self) -> Dict[str, Any]:
        """Returns a dictionary with all detected information."""
        return {
            "effective_cores": self.effective_cores,
            "detection_method": self.detection_method,
            "os_logical_cores": self.os_logical_cores,
            "os_physical_cores": self.os_physical_cores,
            "os_sched_affinity_cores": self.os_sched_affinity_cores,
            "cgroup_type": self.cgroup_type,
            "cgroup_quota_cores": self.cgroup_quota_cores,
            "cgroup_period_us": self.cgroup_period_us,
            "cgroup_shares": self.cgroup_shares,
            "cgroup_usage_total_us": self.cgroup_usage_total_us,
            "cgroup_usage_percpu_us": self.cgroup_usage_percpu_us,
            "platform": platform.system(),
        }
