# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class SingletonStageMixin:
    """
    Marker mixin indicating this Ray actor stage must run as a singleton
    (i.e., only one replica should be instantiated across the system).

    This is used to enforce topology constraints for stages like
    reducers, accumulators, or final sinks where duplication would
    result in incorrect behavior.
    """

    @property
    def is_singleton_stage(self) -> bool:
        return True
