"""Coherent multi-input Mach-Zehnder-interferometer primitives."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import nonnegative_parameter, require_complex_field


class FourInputMZI(nn.Module):
    """Ideal four-input, one-observed-output coherent MZI combiner.

    Input shape is ``(..., 4, samples)``. The observed output port is
    ``Eout = sqrt(eta) / 2 * sum_k(Ek exp(j phi_k))``. The factor 1/2 is the
    unitary four-way-combiner normalization: one active input yields one fourth
    of its power in this output port; four equal in-phase inputs yield all four
    input powers at this port. Other physical output ports are not represented.
    """

    def __init__(self, phase_rad: Tensor | None = None, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        initial_phase = torch.zeros(4) if phase_rad is None else torch.as_tensor(phase_rad, dtype=torch.float32)
        if initial_phase.shape != (4,):
            raise ValueError("phase_rad must have shape (4,)")
        self.phase_rad = nn.Parameter(initial_phase.clone(), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def forward(self, fields: Tensor) -> Tensor:
        require_complex_field(fields)
        if fields.ndim < 2 or fields.shape[-2] != 4:
            raise ValueError("fields must have shape (..., 4, samples)")
        phase = self.phase_rad.to(device=fields.device, dtype=fields.real.dtype)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=fields.device, dtype=fields.real.dtype)
        coefficients = torch.polar(torch.ones_like(phase), phase).view(*([1] * (fields.ndim - 2)), 4, 1)
        ten = torch.tensor(10.0, device=fields.device, dtype=fields.real.dtype)
        amplitude = torch.pow(ten, -loss_db / 20.0)
        return amplitude * (fields * coefficients).sum(dim=-2) / 2.0
