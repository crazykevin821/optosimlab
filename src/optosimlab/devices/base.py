"""Internal validation helpers shared by optical devices."""

from __future__ import annotations

import torch
from torch import Tensor


def require_complex_field(field: Tensor) -> None:
    """Reject ambiguous real-valued optical fields at the module boundary."""
    if not isinstance(field, Tensor):
        raise TypeError("field must be a torch.Tensor")
    if not torch.is_complex(field):
        raise TypeError("field must use a complex dtype (complex64 or complex128)")
    if field.ndim < 1:
        raise ValueError("field must contain a final sample dimension")


def nonnegative_parameter(value: Tensor, name: str, *, floor: float = 0.0) -> Tensor:
    """Validate a physical parameter while preserving its autograd connection."""
    if torch.any(value.detach() < 0):
        raise ValueError(f"{name} must not be negative")
    return value.clamp_min(floor)
