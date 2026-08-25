"""Linear time-invariant frequency-domain filters."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field


class FrequencyDomainLowPass(nn.Module):
    """Zero-phase Butterworth-magnitude low-pass filter.

    ``cutoff_hz`` is the -3 dB *power* cutoff. The field transfer is
    ``H(f) = [1 + (|f| / fc) ** (2 n)] ** (-1/2)``. Its zero phase is a
    deliberate offline-waveform choice; it is not a causal circuit model.
    """

    def __init__(self, grid: SimulationGrid, cutoff_hz: float, order: int = 1, *, trainable: bool = False) -> None:
        super().__init__()
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        if order < 1:
            raise ValueError("order must be at least 1")
        self.grid = grid
        self.order = int(order)
        self.cutoff_hz = nn.Parameter(torch.tensor(float(cutoff_hz)), requires_grad=trainable)

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return real field-amplitude gain in native FFT order."""
        floor = torch.finfo(dtype).eps
        cutoff = nonnegative_parameter(self.cutoff_hz, "cutoff_hz", floor=floor).to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        return (1.0 + (frequency.abs() / cutoff).pow(2 * self.order)).rsqrt()

    def forward(self, field: Tensor) -> Tensor:
        require_complex_field(field)
        response = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.fft(field, dim=-1) * response, dim=-1)
