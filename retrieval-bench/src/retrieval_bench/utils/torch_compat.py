# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Compatibility shims for mismatched torch/transformers versions.

We occasionally update `transformers` on systems where `torch` is older (or vice versa).
Some transformers versions call `torch.is_autocast_enabled(device_type)` while older
torch versions only support `torch.is_autocast_enabled()` (no args).
"""

from typing import Any


def patch_torch_is_autocast_enabled() -> None:
    """
    Patch `torch.is_autocast_enabled` to accept optional args on older torch.

    Safe on newer torch: no-op.
    Safe when torch is unavailable: no-op.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return

    fn = getattr(torch, "is_autocast_enabled", None)
    if not callable(fn):
        return

    # Newer torch accepts a device_type argument (e.g. "cuda").
    # Older torch raises: "takes no arguments (1 given)".
    try:
        _ = fn("cuda")
        return  # already compatible
    except TypeError:
        pass
    except Exception:
        # If it failed for some other reason, don't patch.
        return

    orig = fn

    def _wrapped(*args: Any, **kwargs: Any) -> bool:
        # Ignore any provided args (device_type) and defer to older torch behavior.
        return bool(orig())

    try:
        setattr(torch, "is_autocast_enabled", _wrapped)
    except Exception:
        return
