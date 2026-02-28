# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import json
import argparse
from typing import Optional, Dict, Any, List
import threading
import os

import pandas as pd
import psutil

try:
    import pynvml  # type: ignore
except Exception:  # NVML is optional
    pynvml = None  # type: ignore
try:
    import docker  # type: ignore
except Exception:  # Docker is optional
    docker = None  # type: ignore
import subprocess

# Add these to your requirements.txt:
# pyarrow>=12.0.0
# fastparquet>=2023.2.0

# --- Helpers to mirror docker stats behavior ---
# Keep names internal to avoid changing external API


def _docker_cpu_percent(stats: dict) -> float:
    """Compute CPU% similar to `docker stats` using precpu_stats.
    Falls back gracefully if fields are missing.
    """
    try:
        cpu_stats = stats.get("cpu_stats", {})
        precpu_stats = stats.get("precpu_stats", {})

        cpu_total = cpu_stats.get("cpu_usage", {}).get("total_usage")
        pre_cpu_total = precpu_stats.get("cpu_usage", {}).get("total_usage")

        system_total = cpu_stats.get("system_cpu_usage")
        pre_system_total = precpu_stats.get("system_cpu_usage")

        if cpu_total is None or pre_cpu_total is None or system_total is None or pre_system_total is None:
            return 0.0

        cpu_delta = cpu_total - pre_cpu_total
        system_delta = system_total - pre_system_total

        # Prefer online_cpus when available (cgroup v2 aware); otherwise percpu length
        online_cpus = cpu_stats.get("online_cpus")
        if not online_cpus:
            percpu = cpu_stats.get("cpu_usage", {}).get("percpu_usage") or []
            online_cpus = len(percpu) if percpu else (psutil.cpu_count() or 1)

        if system_delta > 0 and cpu_delta > 0:
            return (cpu_delta / system_delta) * online_cpus * 100.0
        return 0.0
    except Exception:
        return 0.0


def _docker_memory_usage_limit_percent(mem_stats: dict):
    """Return (used_bytes, limit_bytes, percent) using docker's approach.
    On cgroup v1: used = usage - cache. On v2: prefer inactive_file subtraction if present.
    """
    try:
        usage = mem_stats.get("usage", 0) or 0
        limit = mem_stats.get("limit", 0) or 0
        stats = mem_stats.get("stats", {}) or {}

        # Prefer inactive_file (cgroup v2) when present; otherwise cache (v1)
        inactive_file = stats.get("inactive_file")
        if inactive_file is None:
            inactive_file = stats.get("total_inactive_file")
        cache = stats.get("cache")

        if inactive_file is not None:
            used = max(usage - inactive_file, 0)
        elif cache is not None:
            used = max(usage - cache, 0)
        else:
            used = usage

        percent = (used / limit * 100.0) if limit and limit > 0 else 0.0
        return used, limit, percent
    except Exception:
        return 0, 0, 0.0


def _aggregate_network_bytes(stats: dict):
    """Sum rx/tx across all interfaces from docker stats JSON."""
    rx = 0
    tx = 0
    try:
        networks = stats.get("networks", {}) or {}
        for _if, vals in networks.items():
            rx += int(vals.get("rx_bytes", 0) or 0)
            tx += int(vals.get("tx_bytes", 0) or 0)
    except Exception:
        pass
    return rx, tx


def _aggregate_blkio_bytes(stats: dict):
    """Sum blkio read/write bytes from docker stats JSON."""
    read = 0
    write = 0
    try:
        entries = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", []) or []
        for e in entries:
            op = (e.get("op") or "").lower()
            val = int(e.get("value", 0) or 0)
            if op == "read":
                read += val
            elif op == "write":
                write += val
    except Exception:
        pass
    return read, write


class BaseCollector:
    def collect(self) -> Dict[str, Any]:  # pragma: no cover - interface
        return {}

    def close(self) -> None:  # pragma: no cover - optional
        pass


class MemoryCollector(BaseCollector):
    def collect(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        return {"sys_total": mem.total, "sys_used": mem.used, "sys_free": mem.free}


class CPUCollector(BaseCollector):
    def __init__(self, percpu: bool = True, interval: Optional[float] = None) -> None:
        self.percpu = percpu
        self.interval = interval

    def collect(self) -> Dict[str, Any]:
        utils = psutil.cpu_percent(percpu=self.percpu, interval=self.interval)
        if self.percpu:
            return {f"cpu_{i}_utilization": v for i, v in enumerate(utils)}
        else:
            return {"cpu_avg_utilization": utils}


class OpenFilesCollector(BaseCollector):
    def __init__(self, use_lsof_fallback: bool = True) -> None:
        self.use_lsof_fallback = use_lsof_fallback

    def collect(self) -> Dict[str, Any]:
        try:
            total_open_files = len(psutil.Process().net_connections())
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    total_open_files += len(proc.open_files())
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if self.use_lsof_fallback:
                try:
                    result = subprocess.run(["lsof", "-n"], capture_output=True, text=True)
                    lsof_count = len(result.stdout.splitlines()) - 1
                    total_open_files = max(total_open_files, lsof_count)
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass

            max_files = 0
            max_files_process = "None"
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    open_count = len(proc.open_files())
                    if open_count > max_files:
                        max_files = open_count
                        max_files_process = f"{proc.name()}({proc.pid})"
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            try:
                with open("/proc/sys/fs/file-max", "r") as f:
                    fd_max = int(f.read().strip())
                with open("/proc/sys/fs/file-nr", "r") as f:
                    fd_used = int(f.read().split()[0])
                fd_percentage = (fd_used / fd_max) * 100 if fd_max > 0 else 0
            except (FileNotFoundError, ValueError, IndexError):
                fd_max = 0
                fd_used = 0
                fd_percentage = 0

            return {
                "total_open_files": total_open_files,
                "max_files_process": max_files_process,
                "max_files_count": max_files,
                "fd_used": fd_used,
                "fd_max": fd_max,
                "fd_usage_percent": fd_percentage,
            }
        except Exception as e:
            print(f"Error getting open files count: {e}")
            return {
                "total_open_files": -1,
                "max_files_process": f"Error: {str(e)}",
                "max_files_count": -1,
                "fd_used": -1,
                "fd_max": -1,
                "fd_usage_percent": -1,
            }


# -------- Process tree/thread inspector (Python equivalent of thread_checker.sh) --------
def get_process_tree_summary(root_pid: int, verbose: bool = False) -> Dict[str, Any]:
    """Return a summary of a process tree rooted at root_pid.

    Provides per-process thread counts and command names, totals, and aggregation by command.
    This mirrors the functionality of thread_checker.sh using psutil.
    """
    result: Dict[str, Any] = {
        "root_pid": root_pid,
        "processes": [],  # list of {pid, ppid, name, threads}
        "totals": {"total_processes": 0, "total_threads": 0},
        "aggregated_by_command": [],  # list of {command, processes, total_threads}
        "verbose": verbose,
    }
    try:
        if root_pid <= 0:
            return result
        try:
            root = psutil.Process(root_pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Fallback: scan process table, locate root by pid and build tree using PPID relationships
            all_infos: Dict[int, Dict[str, Any]] = {}
            try:
                for p in psutil.process_iter(attrs=["pid", "ppid", "name", "num_threads"]):
                    info = p.info
                    all_infos[info.get("pid")] = {
                        "pid": info.get("pid"),
                        "ppid": info.get("ppid"),
                        "name": info.get("name") or "(unknown)",
                        "threads": int(info.get("num_threads") or 0),
                    }
            except Exception:
                pass
            if root_pid not in all_infos:
                return result
            # Build children map
            by_ppid: Dict[Optional[int], list] = {}
            for it in all_infos.values():
                by_ppid.setdefault(it.get("ppid"), []).append(it)
            # DFS from root_pid to collect subtree
            stack = [root_pid]
            per_pid = []
            total_threads = 0
            seen = set()
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                ent = all_infos.get(cur)
                if not ent:
                    continue
                per_pid.append(ent)
                total_threads += ent.get("threads", 0)
                for child in by_ppid.get(cur, []):
                    cid = child.get("pid")
                    if cid is not None:
                        stack.append(cid)
            result["processes"] = sorted(per_pid, key=lambda x: (x.get("ppid") or -1, x.get("pid") or -1))
            result["totals"] = {"total_processes": len(per_pid), "total_threads": total_threads}
            # Aggregate by command
            agg: Dict[str, Dict[str, int]] = {}
            for it in per_pid:
                cmd = it.get("name") or "(unknown)"
                ent = agg.setdefault(cmd, {"processes": 0, "total_threads": 0})
                ent["processes"] += 1
                ent["total_threads"] += int(it.get("threads") or 0)
            result["aggregated_by_command"] = [
                {"command": k, **v} for k, v in sorted(agg.items(), key=lambda kv: kv[1]["total_threads"], reverse=True)
            ]
            return result

        # Gather all processes in the tree (root + recursive children)
        procs = [root]
        try:
            procs.extend(root.children(recursive=True))
        except Exception:
            pass

        per_pid = []
        total_threads = 0
        for p in procs:
            if p is None:
                continue
            pid = None
            ppid = None
            name = None
            threads = 0
            try:
                pid = p.pid
            except Exception:
                continue
            try:
                ppid = p.ppid()
            except Exception:
                ppid = None
            try:
                name = p.name()
            except Exception:
                name = "(access-denied)"
            try:
                threads = int(p.num_threads())
            except Exception:
                # If threads cannot be read due to permissions, treat as 0 but still include the process
                threads = 0
            info = {"pid": pid, "ppid": ppid, "name": name, "threads": threads}
            per_pid.append(info)
            total_threads += threads

        result["processes"] = sorted(per_pid, key=lambda x: x["pid"])
        result["totals"] = {"total_processes": len(per_pid), "total_threads": total_threads}

        # Aggregate by command
        agg: Dict[str, Dict[str, int]] = {}
        for it in per_pid:
            cmd = it["name"] or "(unknown)"
            ent = agg.setdefault(cmd, {"processes": 0, "total_threads": 0})
            ent["processes"] += 1
            ent["total_threads"] += it["threads"]
        result["aggregated_by_command"] = [
            {"command": k, **v} for k, v in sorted(agg.items(), key=lambda kv: kv[1]["total_threads"], reverse=True)
        ]
    except Exception as e:
        result["error"] = str(e)
    return result


class DiskIOCollector(BaseCollector):
    def collect(self) -> Dict[str, Any]:
        try:
            io_counters = psutil.disk_io_counters()
            return {
                "disk_read_bytes": io_counters.read_bytes,
                "disk_write_bytes": io_counters.write_bytes,
                "disk_read_count": io_counters.read_count,
                "disk_write_count": io_counters.write_count,
                "disk_busy_time": io_counters.busy_time if hasattr(io_counters, "busy_time") else 0,
            }
        except Exception as e:
            print(f"Error getting disk I/O stats: {e}")
            return {
                "disk_read_bytes": -1,
                "disk_write_bytes": -1,
                "disk_read_count": -1,
                "disk_write_count": -1,
                "disk_busy_time": -1,
            }


class NetworkCollector(BaseCollector):
    def collect(self) -> Dict[str, Any]:
        try:
            net_io = psutil.net_io_counters()
            return {
                "net_bytes_sent": net_io.bytes_sent,
                "net_bytes_recv": net_io.bytes_recv,
                "net_packets_sent": net_io.packets_sent,
                "net_packets_recv": net_io.packets_recv,
                "net_errin": net_io.errin,
                "net_errout": net_io.errout,
                "net_dropin": net_io.dropin,
                "net_dropout": net_io.dropout,
            }
        except Exception as e:
            print(f"Error getting network stats: {e}")
            return {
                "net_bytes_sent": -1,
                "net_bytes_recv": -1,
                "net_packets_sent": -1,
                "net_packets_recv": -1,
                "net_errin": -1,
                "net_errout": -1,
                "net_dropin": -1,
                "net_dropout": -1,
            }


class GPUCollector(BaseCollector):
    def __init__(self) -> None:
        self._inited = False
        self._available = False

    def _init(self):
        if self._inited:
            return
        try:
            pynvml.nvmlInit()
            self._available = True
        except Exception as e:
            print(f"GPU monitoring not available: {e}")
            self._available = False
        finally:
            self._inited = True

    def collect(self) -> Dict[str, Any]:
        self._init()
        gpu_stats: Dict[str, Any] = {}
        if not self._available:
            return gpu_stats
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_stats[f"gpu_{i}_total"] = memory_info.total
                    gpu_stats[f"gpu_{i}_used"] = memory_info.used
                    gpu_stats[f"gpu_{i}_free"] = memory_info.free
                    gpu_stats[f"gpu_{i}_utilization"] = utilization.gpu
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_stats[f"gpu_{i}_temp"] = temp
                    except:  # noqa: E722
                        pass
                except pynvml.NVMLError as e:
                    print(f"Error retrieving info for GPU {i}: {e}")
        except Exception as e:
            print(f"Error initializing GPU monitoring: {e}")
        return gpu_stats

    def close(self) -> None:
        if self._inited and self._available:
            try:
                pynvml.nvmlShutdown()
            except:  # noqa: E722
                pass
            finally:
                self._inited = False
                self._available = False


class ProcessThreadCollector(BaseCollector):
    def collect(self) -> Dict[str, Any]:
        proc_count = 0
        thread_count = 0
        try:
            for proc in psutil.process_iter(["pid"]):
                proc_count += 1
                try:
                    thread_count += proc.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            print(f"Error counting processes/threads: {e}")
        return {"system_process_count": proc_count, "system_thread_count": thread_count}


class DockerCollector(BaseCollector):
    """Collect per-container stats and flatten into the row namespace.

    Key format (clear, consistent):
      <prefix>_<container>_<metric>
    Example: docker_nginx_cpu_percent, docker_postgres_mem_used_bytes
    """

    def __init__(
        self,
        client: Optional["docker.DockerClient"] = None,
        key_prefix: str = "docker",
        separator: str = "_",
    ) -> None:
        self.client = client
        self.key_prefix = key_prefix
        self.separator = separator
        if self.client is None:
            try:
                self.client = docker.from_env()
            except Exception as e:
                print("Error connecting to Docker daemon:", e)
                self.client = None

    def collect(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.client is None:
            return out
        try:
            containers = self.client.containers.list()
            for container in containers:
                try:
                    stats_raw = container.stats(stream=False)
                    stats = (
                        json.loads(stats_raw.decode("utf-8"))
                        if isinstance(stats_raw, (bytes, bytearray))
                        else stats_raw
                    )

                    cpu_percent = _docker_cpu_percent(stats)
                    used_bytes, limit_bytes, mem_percent = _docker_memory_usage_limit_percent(
                        stats.get("memory_stats", {})
                    )
                    mem_usage_gb = used_bytes / (1024**3)  # noqa: F841
                    mem_limit_gb = limit_bytes / (1024**3) if limit_bytes else 0  # noqa: F841

                    rx_bytes, tx_bytes = _aggregate_network_bytes(stats)
                    blk_read, blk_write = _aggregate_blkio_bytes(stats)

                    # Best-effort open files from container init PID
                    try:
                        inspect_data = container.attrs
                        pid = inspect_data.get("State", {}).get("Pid", 0)
                        if pid and pid > 0:
                            proc = psutil.Process(pid)
                            open_files_count = len(proc.open_files())
                        else:
                            open_files_count = -1
                    except Exception:
                        open_files_count = -1

                    cname = container.name
                    sep = self.separator
                    pref = f"{self.key_prefix}{sep}{cname}" if self.key_prefix else cname
                    out.update(
                        {
                            f"{pref}{sep}cpu_percent": cpu_percent,
                            # memory (expose both raw bytes and derived percent/limit)
                            f"{pref}{sep}mem_used_bytes": int(used_bytes),
                            f"{pref}{sep}mem_limit_bytes": int(limit_bytes),
                            f"{pref}{sep}mem_percent": mem_percent,
                            # open files (best-effort)
                            f"{pref}{sep}open_files": open_files_count,
                            # cumulative counters for per-second derivation
                            f"{pref}{sep}net_rx_bytes": rx_bytes,
                            f"{pref}{sep}net_tx_bytes": tx_bytes,
                            f"{pref}{sep}blkio_read_bytes": blk_read,
                            f"{pref}{sep}blkio_write_bytes": blk_write,
                        }
                    )
                except Exception as e:
                    print(f"Error retrieving stats for container {container.name}: {e}")
        except Exception as e:
            print("Error listing Docker containers:", e)
        return out


def calculate_deltas(current, previous, delta_keys):
    deltas = {}
    if previous:
        for key in delta_keys:
            if key in current and key in previous:
                if isinstance(current[key], (int, float)) and isinstance(previous[key], (int, float)):
                    time_diff = (current["timestamp"] - previous["timestamp"]).total_seconds()
                    if time_diff > 0:
                        delta_per_sec = (current[key] - previous[key]) / time_diff
                        deltas[f"{key}_per_sec"] = delta_per_sec
    return deltas


class SystemTracer:
    """Encapsulated system monitoring with configurable options.

    Provides collection of system metrics, optional Docker and GPU stats,
    delta computation for cumulative counters, and periodic Parquet writing.
    """

    def __init__(
        self,
        sample_interval: float = 5.0,
        write_interval: float = 10.0,
        output_file: str = "system_monitor.parquet",
        enable_gpu: bool = True,
        enable_docker: bool = True,
        docker_client: Optional["docker.DockerClient"] = None,
        collectors: Optional[List[BaseCollector]] = None,
        use_utc: bool = False,
        write_final: bool = True,
    ) -> None:
        self.sample_interval = sample_interval
        self.write_interval = write_interval
        self.output_file = output_file
        self.enable_gpu = enable_gpu
        self.enable_docker = enable_docker
        self.data_buffer: List[Dict[str, Any]] = []
        self.previous_row: Optional[Dict[str, Any]] = None
        self.delta_keys: List[str] = [
            "disk_read_bytes",
            "disk_write_bytes",
            "disk_read_count",
            "disk_write_count",
            "net_bytes_sent",
            "net_bytes_recv",
            "net_packets_sent",
            "net_packets_recv",
        ]
        self.last_write_time = time.time()
        self.docker_client = docker_client
        self.use_utc = use_utc
        self.write_final = write_final
        if self.enable_docker and self.docker_client is None:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                print(f"Docker client not available: {e}")
                self.docker_client = None

        # Initialize per-metric collectors (allow override)
        if collectors is not None:
            self.collectors = collectors
        else:
            self.collectors: List[BaseCollector] = [
                MemoryCollector(),
                CPUCollector(percpu=True, interval=None),
                OpenFilesCollector(),
                DiskIOCollector(),
                NetworkCollector(),
                ProcessThreadCollector(),
            ]
        self.gpu_collector: Optional[GPUCollector] = None
        if self.enable_gpu:
            self.gpu_collector = GPUCollector()
            self.collectors.append(self.gpu_collector)
        self.docker_collector: Optional[DockerCollector] = None
        if self.enable_docker and self.docker_client is not None:
            self.docker_collector = DockerCollector(client=self.docker_client)
            self.collectors.append(self.docker_collector)
        # Control flags/state
        self._stop_event = threading.Event()

    def _shutdown_gpu(self) -> None:
        # Back-compat: close GPU collector if present
        if self.gpu_collector is not None:
            try:
                self.gpu_collector.close()
            except Exception:
                pass

    def write_parquet_to(self, df: pd.DataFrame, destination_path: str) -> None:
        """Atomically write the dataframe to a specific parquet destination path.

        Mirrors write_parquet but targets the provided destination rather than self.output_file.
        """
        tmp_path = f"{destination_path}.tmp"
        # Try pyarrow first
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pandas(df)
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, destination_path)
            return
        except ImportError:
            pass
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

        # Fallback: fastparquet via pandas
        try:
            df.to_parquet(tmp_path, engine="fastparquet")
            os.replace(tmp_path, destination_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def collect_once(self) -> Dict[str, Any]:
        """Collect a single snapshot of system metrics, computing deltas if possible."""
        timestamp = pd.Timestamp.utcnow() if self.use_utc else pd.Timestamp.now()
        row: Dict[str, Any] = {"timestamp": timestamp}
        # Aggregate from all collectors
        for collector in self.collectors:
            try:
                data = collector.collect()
                if not data:
                    continue
                row.update(data)
            except Exception as e:
                print(f"Collector {collector.__class__.__name__} failed: {e}")

        # If Docker collector present, add its cumulative keys to delta set
        if self.docker_collector is not None:
            # Match new naming: docker_<container>_<metric>
            suffixes = (
                "_net_rx_bytes",
                "_net_tx_bytes",
                "_blkio_read_bytes",
                "_blkio_write_bytes",
            )
            for k in list(row.keys()):
                if any(k.endswith(sfx) for sfx in suffixes) and k not in self.delta_keys:
                    self.delta_keys.append(k)

        # Deltas
        if self.previous_row:
            deltas = calculate_deltas(row, self.previous_row, self.delta_keys)
            row.update(deltas)
        self.previous_row = row.copy()
        return row

    def write_parquet(self, df: pd.DataFrame) -> None:
        """Atomically write the current dataframe to the parquet output path.

        Prefers pyarrow; falls back to fastparquet if available. Writes to a temp
        file and atomically replaces the target so readers never see partial data.
        """
        tmp_path = f"{self.output_file}.tmp"
        # Try pyarrow first
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            table = pa.Table.from_pandas(df)
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, self.output_file)
            return
        except ImportError:
            pass
        except Exception:
            # If pyarrow present but write failed, clean up and re-raise to try fallback
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

        # Fallback: fastparquet via pandas
        try:
            df.to_parquet(tmp_path, engine="fastparquet")
            os.replace(tmp_path, self.output_file)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def run(self, duration: Optional[float] = None, verbose: bool = True) -> None:
        """Run monitoring loop. If duration is set (seconds), stop after duration; else run until Ctrl+C."""
        start_time = time.time()
        if verbose:
            print(
                f"Starting system monitoring. Data will be written to {self.output_file} every "
                f"{self.write_interval} seconds."
            )
            print("Press Ctrl+C to stop monitoring." if duration is None else f"Stopping after {duration} seconds...")
        try:
            while not self._stop_event.is_set():
                row = self.collect_once()
                self.data_buffer.append(row)

                if verbose:
                    ts = row["timestamp"]
                    cpu_cols = [k for k in row.keys() if k.startswith("cpu_") and k.endswith("_utilization")]
                    cpu_vals = [row[k] for k in cpu_cols]
                    cpu_avg = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
                    mem_pct = (row["sys_used"] / row["sys_total"] * 100.0) if row.get("sys_total") else 0.0
                    print(
                        f"\n[{ts}] CPU Avg: {cpu_avg:.1f}% | Memory: {mem_pct:.1f}% | "
                        f"Open Files: {row.get('total_open_files', 0)}"
                    )
                    if "disk_read_bytes_per_sec" in row:
                        print(
                            f"Disk I/O: {row['disk_read_bytes_per_sec']/1024**2:.2f} MB/s read, "
                            f"{row['disk_write_bytes_per_sec']/1024**2:.2f} MB/s write"
                        )
                    if "net_bytes_recv_per_sec" in row:
                        print(
                            f"Network: {row['net_bytes_recv_per_sec']/1024**2:.2f} MB/s down, "
                            f"{row['net_bytes_sent_per_sec']/1024**2:.2f} MB/s up"
                        )

                # Periodic write (overwrite full buffered data) if enabled
                now = time.time()
                if (
                    self.write_interval
                    and self.write_interval > 0
                    and (now - self.last_write_time) >= self.write_interval
                ):
                    df = pd.DataFrame(self.data_buffer)
                    try:
                        self.write_parquet(df)
                        if verbose:
                            print(
                                f"Total accumulated data ({len(self.data_buffer)} rows) "
                                f"written to {self.output_file} at {row['timestamp']}"
                            )
                        self.last_write_time = now
                    except Exception as e:
                        print(f"Error writing periodic data: {e}")

                # Stop conditions
                if duration is not None and (now - start_time) >= duration:
                    break

                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            if verbose:
                print("\nStopping monitoring. Writing final data batch...")
        finally:
            # Final write if enabled
            if self.write_final and self.data_buffer:
                df = pd.DataFrame(self.data_buffer)
                try:
                    self.write_parquet(df)
                    if verbose:
                        print(f"Final data written to {self.output_file}. Exiting.")
                except Exception as e:
                    print(f"Error writing final data: {e}")
            # Close any collectors that need cleanup
            try:
                self._shutdown_gpu()
            except Exception:
                pass

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._stop_event.set()

    def reset(self) -> None:
        """Clear accumulated data and deltas. Does not change output file."""
        self.data_buffer = []
        self.previous_row = None
        self.last_write_time = time.time()
        # Do not clear stop flag to allow caller to decide lifecycle

    def set_output_file(self, output_file: str) -> None:
        """Update the output parquet path used by periodic writes."""
        self.output_file = output_file

    def snapshot(self, output_file: Optional[str] = None) -> str:
        """Write the current buffered dataframe to the specified parquet path (or self.output_file).

        Returns the path written.
        """
        base_path = output_file or self.output_file
        if not base_path:
            raise ValueError("No output_file specified for snapshot.")
        # If destination exists, create a unique suffixed name: file.parquet -> file_0.parquet, ...
        root, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".parquet"
            root = base_path  # original base without extension
            base_path = base_path + ext

        path = base_path
        if os.path.exists(path):
            idx = 0
            while True:
                candidate = f"{root}_{idx}{ext}"
                if not os.path.exists(candidate):
                    path = candidate
                    break
                idx += 1
        df = pd.DataFrame(self.data_buffer)
        if not df.empty:
            self.write_parquet_to(df, path)
        else:
            # Still write an empty table with schema
            self.write_parquet_to(pd.DataFrame([]), path)
        return path


# -------- Functional API --------
def collect_system_snapshot(enable_gpu: bool = True, enable_docker: bool = True, docker_client=None) -> Dict[str, Any]:
    tracer = SystemTracer(
        sample_interval=0.0,
        write_interval=0.0,
        output_file="",
        enable_gpu=enable_gpu,
        enable_docker=enable_docker,
        docker_client=docker_client,
        use_utc=False,
    )
    return tracer.collect_once()


def monitor_to_parquet(
    output_file: str = "system_monitor.parquet",
    sample_interval: float = 5.0,
    write_interval: float = 10.0,
    duration: Optional[float] = None,
    enable_gpu: bool = True,
    enable_docker: bool = True,
    docker_client=None,
    verbose: bool = True,
    use_utc: bool = False,
) -> None:
    tracer = SystemTracer(
        sample_interval=sample_interval,
        write_interval=write_interval,
        output_file=output_file,
        enable_gpu=enable_gpu,
        enable_docker=enable_docker,
        docker_client=docker_client,
        use_utc=use_utc,
    )
    tracer.run(duration=duration, verbose=verbose)


# -------- CLI utility --------
def main():
    parser = argparse.ArgumentParser(description="System monitor/tracer CLI")
    sub = parser.add_subparsers(dest="command")

    # run (default)
    p_run = sub.add_parser("run", help="Run continuous monitoring and write Parquet")
    p_run.add_argument("--output", default="system_monitor.parquet", help="Parquet output file path")
    p_run.add_argument("--sample-interval", type=float, default=5.0, help="Sampling interval seconds")
    p_run.add_argument("--write-interval", type=float, default=10.0, help="Write interval seconds")
    p_run.add_argument("--duration", type=float, default=None, help="Optional duration to run (seconds)")
    p_run.add_argument("--no-gpu", action="store_true", help="Disable GPU collection")
    p_run.add_argument("--no-docker", action="store_true", help="Disable Docker collection")
    p_run.add_argument("--quiet", action="store_true", help="Reduce console output")
    p_run.add_argument("--utc", action="store_true", help="Record timestamps in UTC (default is local time)")

    # snapshot
    p_snap = sub.add_parser("snapshot", help="Collect a single snapshot and print JSON")
    p_snap.add_argument("--no-gpu", action="store_true", help="Disable GPU collection")
    p_snap.add_argument("--no-docker", action="store_true", help="Disable Docker collection")
    p_snap.add_argument("--utc", action="store_true", help="Use UTC timestamp for the snapshot")

    # proctree (process/thread inspection)
    p_tree = sub.add_parser("proctree", help="Inspect a process tree and summarize threads")
    p_tree.add_argument("pid", type=int, help="Root PID to inspect")
    p_tree.add_argument("--verbose", action="store_true", help="Verbose per-PID output in JSON")

    args = parser.parse_args()
    if not getattr(args, "command", None):
        # No subcommand provided; default to 'run' so that subparser defaults are applied
        args = parser.parse_args(["run"])
    cmd = args.command

    if cmd == "snapshot":
        # One-off snapshot; use_utc affects only the timestamp on this row
        tracer = SystemTracer(
            sample_interval=0.0,
            write_interval=0.0,
            output_file="",
            enable_gpu=not args.no_gpu,
            enable_docker=not args.no_docker,
            docker_client=None,
            use_utc=bool(getattr(args, "utc", False)),
        )
        snap = tracer.collect_once()
        print(json.dumps(snap, default=str))
        return

    if cmd == "proctree":
        summary = get_process_tree_summary(args.pid, verbose=args.verbose)
        print(json.dumps(summary, default=str))
        return

    # default: run
    monitor_to_parquet(
        output_file=args.output,
        sample_interval=args.sample_interval,
        write_interval=args.write_interval,
        duration=args.duration,
        enable_gpu=not args.no_gpu,
        enable_docker=not args.no_docker,
        verbose=not args.quiet,
        use_utc=bool(getattr(args, "utc", False)),
    )


if __name__ == "__main__":
    main()
