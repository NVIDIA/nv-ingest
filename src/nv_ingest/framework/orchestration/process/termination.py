# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Process termination utilities, isolated to avoid circular imports.

This module provides functions to terminate a process and its entire process
group safely, without depending on pipeline construction or Ray types.
"""

import logging
import os
import signal
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _safe_log(level: int, msg: str) -> None:
    """Best-effort logging that won't crash during interpreter shutdown."""
    try:
        logger.log(level, msg)
    except Exception:
        try:
            # Fallback to stderr if available
            import sys

            if hasattr(sys, "__stderr__") and sys.__stderr__:
                sys.__stderr__.write(msg + "\n")
                sys.__stderr__.flush()
        except Exception:
            pass


def kill_pipeline_process_group(process) -> None:
    """
    Kill a process and its entire process group.

    Accepts either a multiprocessing.Process-like object exposing a ``pid`` attribute
    or a raw PID integer. Sends SIGTERM to the process group first, and escalates
    to SIGKILL if it does not terminate within a short grace period.

    Parameters
    ----------
    process : multiprocessing.Process | int
        Process handle (or a raw PID int) for the process whose process group should be terminated.
    """
    proc: Optional[object] = None
    pid: Optional[int] = None

    if isinstance(process, int):
        pid = process
    elif hasattr(process, "pid"):
        proc = process
        try:
            pid = int(getattr(proc, "pid"))
        except Exception as e:
            raise AttributeError(f"Invalid process-like object without usable pid: {e}")
    else:
        raise AttributeError(
            "kill_pipeline_process_group expects a multiprocessing.Process or a PID int (process-like object with .pid)"
        )

    if proc is not None and hasattr(proc, "is_alive") and not proc.is_alive():
        _safe_log(logging.DEBUG, "Process already terminated")
        return

    if pid is None:
        raise AttributeError("Unable to determine PID for process group termination")

    _safe_log(logging.INFO, f"Terminating pipeline process group (PID: {pid})")

    try:
        # Send graceful termination to the entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)

        # If we have a Process handle, give it a chance to exit cleanly
        if proc is not None and hasattr(proc, "join"):
            try:
                proc.join(timeout=5.0)
            except Exception:
                pass
            still_alive = getattr(proc, "is_alive", lambda: True)()
        else:
            # Without a handle, provide a small grace period
            time.sleep(2.0)
            try:
                _ = os.getpgid(pid)
                still_alive = True
            except Exception:
                still_alive = False

        if still_alive:
            _safe_log(logging.WARNING, "Process group did not terminate gracefully, using SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            finally:
                if proc is not None and hasattr(proc, "join"):
                    try:
                        proc.join(timeout=3.0)
                    except Exception:
                        pass

    except (ProcessLookupError, OSError) as e:
        _safe_log(logging.DEBUG, f"Process group already terminated or not found: {e}")
