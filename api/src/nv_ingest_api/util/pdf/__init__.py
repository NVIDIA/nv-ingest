# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.util.pdf.pdfium import (
    _compute_render_scale_to_fit,
    convert_bitmap_to_corrected_numpy,
    is_scanned_page,
)
