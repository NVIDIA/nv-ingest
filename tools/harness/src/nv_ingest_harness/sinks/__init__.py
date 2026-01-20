# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Sinks for pluggable result processing."""

from nv_ingest_harness.sinks.base import Sink
from nv_ingest_harness.sinks.slack import SlackSink
from nv_ingest_harness.sinks.history import HistorySink

__all__ = ["Sink", "SlackSink", "HistorySink"]
