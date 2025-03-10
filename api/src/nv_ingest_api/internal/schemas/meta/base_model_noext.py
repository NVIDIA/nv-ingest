# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import ConfigDict, BaseModel


# Define a base class with extra fields forbidden
class BaseModelNoExt(BaseModel):
    model_config = ConfigDict(extra="forbid")
