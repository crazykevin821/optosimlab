"""Deterministic optical frequency-comb source on an FFT sampling grid."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter


class OpticalFrequencyComb(nn.Module):
    """Generate a carrier-centred, discrete optical frequency comb.

    The source contains an odd number of lines at offsets
    ``k * line_spacing_hz`` where ``k = -(line_count//2), ..., +(line_count//2)``.
    Each line has trainable power and phase.  A line must fall exactly on an FFT
    bin for the requested record length; this deterministic first version
    rejects off-bin spacings rather than silently introducing spectral leakage.

    If ``P_k`` is a line power and ``phi_k`` its phase, the unnormalised DFT
    coefficient is ``N sqrt(P_k) exp(j phi_k)``.  PyTorch's inverse FFT then
    yields a waveform whose mean power is exactly ``sum(P_k)`` for distinct
    bins.  Frequencies are offsets from the optical carrier.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        line_spacing_hz: float,
        line_count: int = 3,
        *,
        line_powers: float | Tensor = 1.0,
        line_phases_rad: Tensor | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if line_spacing_hz <= 0:
            raise ValueError("line_spacing_hz must be positive")
        if line_count < 1 or line_count % 2 == 0:
            raise ValueError("line_count must be a positive odd integer so one line is at the carrier")
        self.grid = grid
        self.line_spacing_hz = float(line_spacing_hz)
        self.line_count = int(line_count)

        powers = torch.as_tensor(line_powers, dtype=torch.float32)
        if powers.ndim == 0:
            powers = powers.expand(line_count).clone()
        if powers.shape != (line_count,):
            raise ValueError("line_powers must be a scalar or a tensor with shape (line_count,)")
        if torch.any(powers < 0):
            raise ValueError("line_powers must not be negative")

        phases = torch.zeros(line_count) if line_phases_rad is None else torch.as_tensor(line_phases_rad, dtype=torch.float32)
        if phases.shape != (line_count,):
            raise ValueError("line_phases_rad must have shape (line_count,)")

        self.line_powers = nn.Parameter(powers, requires_grad=trainable)
        self.line_phases_rad = nn.Parameter(phases.clone(), requires_grad=trainable)

    @property
    def total_power(self) -> Tensor:
        """The sum of configured per-line powers in field-squared units."""
        return nonnegative_parameter(self.line_powers, "line_powers").sum()

    def line_bin_indices(self, samples: int) -> Tensor:
        """Return native-FFT indices after validating bin alignment and bandwidth."""
        if samples <= 0:
            raise ValueError("samples must be positive")
        bin_width_hz = self.grid.sample_rate_hz / samples
        spacing_in_bins = self.line_spacing_hz / bin_width_hz
        integer_spacing = round(spacing_in_bins)
        if not math.isclose(spacing_in_bins, integer_spacing, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(
                "line_spacing_hz must be an integer multiple of sample_rate_hz / samples; "
                "choose an aligned record length or implement an interpolation model"
            )
        offsets = torch.arange(-(self.line_count // 2), self.line_count // 2 + 1, dtype=torch.long)
        signed_bins = offsets * integer_spacing
        if int(signed_bins.abs().max()) > (samples - 1) // 2:
            raise ValueError("comb lines exceed the representable positive/negative frequency range")
        return signed_bins.remainder(samples)

    def line_offsets_hz(self, samples: int, *, device: torch.device | None = None) -> Tensor:
        """Carrier-relative offsets in the same order as the power and phase parameters."""
        indices = self.line_bin_indices(samples).to(device=device)
        signed = torch.where(indices <= samples // 2, indices, indices - samples)
        return signed.to(torch.float64) * (self.grid.sample_rate_hz / samples)

    def spectrum(self, samples: int, *, device: torch.device | None = None, real_dtype: torch.dtype | None = None) -> Tensor:
        """Return the unnormalised FFT spectrum of the comb waveform."""
        indices = self.line_bin_indices(samples)
        device = device or self.line_powers.device
        dtype = real_dtype or self.line_powers.dtype
        powers = nonnegative_parameter(self.line_powers, "line_powers").to(device=device, dtype=dtype)
        phases = self.line_phases_rad.to(device=device, dtype=dtype)
        coefficients = torch.polar(samples * torch.sqrt(powers), phases)
        base = torch.zeros(samples, device=device, dtype=coefficients.dtype)
        return base.scatter(0, indices.to(device=device), coefficients)

    def forward(self, samples: int, *, device: torch.device | None = None, real_dtype: torch.dtype | None = None) -> Tensor:
        """Generate a one-dimensional complex envelope with ``samples`` points."""
        return torch.fft.ifft(self.spectrum(samples, device=device, real_dtype=real_dtype))
