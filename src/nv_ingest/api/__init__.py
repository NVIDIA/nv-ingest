# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nv_ingest.api package."""

from .tracing import traced_endpoint  # re-export for convenience

__all__ = ["traced_endpoint"]
