# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings


# Suppressing CUDA-related warnings when running NV-Ingest on a CPU-only system.
#
# The warnings originate from Numba, which attempts to initialize CUDA even if no GPU is available.
# These warnings include errors about missing CUDA drivers or failing to dlopen `libcuda.so.1`.
#
# By temporarily ignoring `UserWarning` during the import, we prevent unnecessary clutter in logs
# while ensuring that cuDF still functions in CPU mode.
#
# Note: This does not affect cuDF behavior - it will still fall back to CPU execution if no GPU is detected.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    # import cudf
    # TODO(Devin) No cudf import in this file, but keeping it here for future use
