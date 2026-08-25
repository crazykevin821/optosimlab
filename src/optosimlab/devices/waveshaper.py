"""Differentiable programmable frequency-domain shaping."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field


class WaveShaper(nn.Module):
    """A trainable WaveShaper-like attenuation and phase mask.

    Control points are ordered on an fft-shifted frequency axis from negative
    offset to positive offset, so the centre control point represents the
    optical carrier.  The response is linearly interpolated onto the current
    FFT grid.  ``attenuation_db`` is constrained to be non-negative, matching
    a passive waveshaper; arbitrary phase is allowed.
    """

    def __init__(self, grid: SimulationGrid, control_points: int = 129, *, trainable: bool = True) -> None:
        super().__init__()
        if control_points < 2:
            raise ValueError("control_points must be at least 2")
        self.grid = grid
        self.control_points = int(control_points)
        self.attenuation_db = nn.Parameter(torch.zeros(control_points), requires_grad=trainable)
        self.phase_rad = nn.Parameter(torch.zeros(control_points), requires_grad=trainable)

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if samples <= 0:
            raise ValueError("samples must be positive")
        attenuation = nonnegative_parameter(self.attenuation_db, "attenuation_db").to(device=device, dtype=dtype)
        phase = self.phase_rad.to(device=device, dtype=dtype)
        attenuation = F.interpolate(attenuation.view(1, 1, -1), size=samples, mode="linear", align_corners=True).view(-1)
        phase = F.interpolate(phase.view(1, 1, -1), size=samples, mode="linear", align_corners=True).view(-1)
        ten = torch.tensor(10.0, device=device, dtype=dtype)
        amplitude = torch.pow(ten, -attenuation / 20.0)
        return torch.polar(amplitude, phase)

    def forward(self, field: Tensor) -> Tensor:
        require_complex_field(field)
        spectrum_shifted = torch.fft.fftshift(torch.fft.fft(field, dim=-1), dim=-1)
        response = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.ifftshift(spectrum_shifted * response, dim=-1), dim=-1)
