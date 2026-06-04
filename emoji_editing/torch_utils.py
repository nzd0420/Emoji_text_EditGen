"""Shared torch helpers (precision/dtype resolution)."""

from __future__ import annotations

import torch


def resolve_dtype(precision: str) -> torch.dtype:
    """Map a precision string to a concrete torch dtype.

    ``"bf16"`` -> bfloat16, ``"fp16"`` -> float16, anything else -> float32.
    """

    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Like :func:`resolve_dtype`, but returns ``None`` for full precision.

    Useful for ``torch.autocast(..., enabled=dtype is not None)``.
    """

    dtype = resolve_dtype(precision)
    return None if dtype is torch.float32 else dtype
