import time
import json
import pandas as pd
import psutil
import pynvml
import docker
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


def get_system_memory():
    mem = psutil.virtual_memory()
    return mem.total, mem.used, mem.free


def get_cpu_utilization():
    return psutil.cpu_percent(percpu=True, interval=None)


def get_open_files_count():
    try:
        # Use net_connections() to avoid deprecation warnings
        total_open_files = len(psutil.Process().net_connections())
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                total_open_files += len(proc.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        try:
            result = subprocess.run(["lsof", "-n"], capture_output=True, text=True)
            lsof_count = len(result.stdout.splitlines()) - 1  # Subtract header line
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


def get_disk_io_stats():
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


def get_network_stats():
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


def get_gpu_stats():
    gpu_stats = {}
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("No GPU devices found.")
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


def get_process_thread_counts():
    """Return a tuple (process_count, thread_count) for the entire system.
    Uses psutil.process_iter with resilient error handling.
    """
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
    return proc_count, thread_count


def get_docker_container_stats(client=None):
    """Collect per-container CPU%, memory usage/limit/%, open files, and cumulative IO counters.
    Uses a single non-streaming stats call per container and Docker-provided precpu_stats.
    """
    container_stats = {}
    try:
        if client is None:
            try:
                client = docker.from_env()
            except Exception as e:
                print("Error connecting to Docker daemon:", e)
                return container_stats

        containers = client.containers.list()
        for container in containers:
            try:
                stats_raw = container.stats(stream=False)
                stats = (
                    json.loads(stats_raw.decode("utf-8")) if isinstance(stats_raw, (bytes, bytearray)) else stats_raw
                )

                cpu_percent = _docker_cpu_percent(stats)
                used_bytes, limit_bytes, mem_percent = _docker_memory_usage_limit_percent(stats.get("memory_stats", {}))
                mem_usage_gb = used_bytes / (1024**3)
                mem_limit_gb = limit_bytes / (1024**3) if limit_bytes else 0

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

                container_stats[container.name] = {
                    "container_cpu_percent": cpu_percent,
                    "container_mem_usage": mem_usage_gb,
                    "container_mem_limit": mem_limit_gb,
                    "container_mem_percent": mem_percent,
                    "container_open_files": open_files_count,
                    # cumulative counters for per-second derivation
                    "container_net_rx_bytes": rx_bytes,
                    "container_net_tx_bytes": tx_bytes,
                    "container_blkio_read_bytes": blk_read,
                    "container_blkio_write_bytes": blk_write,
                }
            except Exception as e:
                print(f"Error retrieving stats for container {container.name}: {e}")
    except Exception as e:
        print("Error listing Docker containers:", e)
    return container_stats


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


def main():
    try:
        pynvml.nvmlInit()
        gpu_available = True
    except Exception as e:
        print(f"GPU monitoring not available: {e}")
        gpu_available = False

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required for parquet output.")

    output_file = "system_monitor.parquet"
    previous_row = None
    # Accumulate all rows in this list (do not clear it after writing)
    data_buffer = []

    delta_keys = [
        "disk_read_bytes",
        "disk_write_bytes",
        "disk_read_count",
        "disk_write_count",
        "net_bytes_sent",
        "net_bytes_recv",
        "net_packets_sent",
        "net_packets_recv",
    ]

    sample_interval = 5
    write_interval = 10
    last_write_time = time.time()

    # Initialize Docker client once (best effort)
    try:
        docker_client = docker.from_env()
    except Exception as e:
        print(f"Docker client not available: {e}")
        docker_client = None

    print(f"Starting system monitoring. Data will be written to {output_file} every {write_interval} seconds.")
    print("Press Ctrl+C to stop monitoring.")

    try:
        while True:
            timestamp = pd.Timestamp.now()
            sys_total, sys_used, sys_free = get_system_memory()
            cpu_utils = get_cpu_utilization()
            files_info = get_open_files_count()
            disk_stats = get_disk_io_stats()
            network_stats = get_network_stats()
            proc_count, thread_count = get_process_thread_counts()

            row = {
                "timestamp": timestamp,
                "sys_total": sys_total,
                "sys_used": sys_used,
                "sys_free": sys_free,
                "system_process_count": proc_count,
                "system_thread_count": thread_count,
            }
            row.update(files_info)
            row.update(disk_stats)
            row.update(network_stats)
            for idx, util in enumerate(cpu_utils):
                row[f"cpu_{idx}_utilization"] = util

            if gpu_available:
                gpu_info = get_gpu_stats()
                row.update(gpu_info)

            try:
                container_stats = get_docker_container_stats(docker_client)
                for container_name, stats in container_stats.items():
                    for key, value in stats.items():
                        row[f"{container_name}_{key}"] = value

                    # Dynamically add cumulative container counters to delta_keys for per-sec rates
                    for k in [
                        "container_net_rx_bytes",
                        "container_net_tx_bytes",
                        "container_blkio_read_bytes",
                        "container_blkio_write_bytes",
                    ]:
                        composed = f"{container_name}_{k}"
                        if composed not in delta_keys:
                            delta_keys.append(composed)
            except Exception as e:
                print(f"Error processing container stats: {e}")

            if previous_row:
                deltas = calculate_deltas(row, previous_row, delta_keys)
                row.update(deltas)

            previous_row = row.copy()
            data_buffer.append(row)

            print(f"\n[{timestamp}] System Monitor Stats:")
            print(
                f"CPU Avg: {sum(cpu_utils)/len(cpu_utils):.1f}% | Memory: {sys_used/sys_total*100:.1f}% | "
                f"Open Files: {files_info['total_open_files']}"
            )
            print(
                f"FD Usage: {files_info['fd_usage_percent']:.2f}% | Process with most files: "
                f"{files_info['max_files_process']} ({files_info['max_files_count']} files)"
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

            # Every write_interval seconds, write the full accumulated dataframe to disk (overwriting the file)
            if time.time() - last_write_time >= write_interval:
                df = pd.DataFrame(data_buffer)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_file)
                print(f"Total accumulated data ({len(data_buffer)} rows) written to {output_file} at {timestamp}")
                last_write_time = time.time()

            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\nStopping monitoring. Writing final data batch...")
        if data_buffer:
            df = pd.DataFrame(data_buffer)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_file)
        print(f"Final data written to {output_file}. Exiting.")

    finally:
        if gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:  # noqa: E722
                pass


if __name__ == "__main__":
    main()
