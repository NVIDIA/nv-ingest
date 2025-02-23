# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.util.exception_handlers.schemas import schema_exception_handler


@schema_exception_handler
def validate_schema(metadata, Schema):
    return Schema(**metadata)
