"""Time-domain electro-optic frequency-comb model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field


class ElectroOpticComb(nn.Module):
    """Cascade ideal phase modulators driven by phase-coherent RF tones.

    Each stage applies the lossless phase-modulation transfer function

    ``E_out(t) = E_in(t) exp(j * beta * sin(2 pi f_rf t + phi_rf))``.

    For a continuous-wave input, the Jacobi-Anger expansion produces comb
    teeth with complex amplitudes ``J_n(beta) exp(j n phi_rf)`` at offsets
    ``n * f_rf``.  The stage transfer has unit magnitude, so cascading stages
    does not change optical power before the optional *total* insertion loss.
    RF frequencies are deliberately structural values rather than parameters:
    they must be integer FFT bins for the finite record to represent a periodic
    comb without spectral leakage.  Modulation indices, RF phases and loss are
    registered ``nn.Parameter`` instances for optimisation.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        modulation_frequency_hz: float | Tensor,
        modulation_indices_rad: float | Tensor = 1.0,
        rf_phases_rad: float | Tensor = 0.0,
        insertion_loss_db: float = 0.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        indices = torch.as_tensor(modulation_indices_rad, dtype=torch.float32)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError("modulation_indices_rad must be a non-empty scalar or one-dimensional tensor")

        frequencies = torch.as_tensor(modulation_frequency_hz, dtype=torch.float64)
        if frequencies.ndim == 0:
            frequencies = frequencies.expand(indices.numel()).clone()
        if frequencies.shape != indices.shape:
            raise ValueError("modulation_frequency_hz must be a scalar or have one entry per modulation stage")
        if torch.any(frequencies <= 0):
            raise ValueError("modulation_frequency_hz must be positive")

        phases = torch.as_tensor(rf_phases_rad, dtype=torch.float32)
        if phases.ndim == 0:
            phases = phases.expand(indices.numel()).clone()
        if phases.shape != indices.shape:
            raise ValueError("rf_phases_rad must be a scalar or have one entry per modulation stage")

        self.grid = grid
        self.register_buffer("modulation_frequencies_hz", frequencies.clone(), persistent=True)
        self.modulation_indices_rad = nn.Parameter(indices.clone(), requires_grad=trainable)
        self.rf_phases_rad = nn.Parameter(phases.clone(), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    @property
    def stage_count(self) -> int:
        """Number of cascaded RF phase-modulation stages."""
        return int(self.modulation_indices_rad.numel())

    def frequency_bin_indices(self, samples: int) -> Tensor:
        """Validate record-periodic RF tones and return their positive FFT bins."""
        if samples <= 1:
            raise ValueError("samples must be greater than one")
        bins = self.modulation_frequencies_hz * samples / self.grid.sample_rate_hz
        rounded = torch.round(bins)
        if not torch.allclose(bins, rounded, rtol=1e-10, atol=1e-8):
            raise ValueError(
                "modulation_frequency_hz must be an integer multiple of sample_rate_hz / samples; "
                "choose an aligned record length"
            )
        if torch.any(rounded >= samples / 2):
            raise ValueError("modulation_frequency_hz must be below the Nyquist frequency")
        return rounded.to(dtype=torch.long)

    def field_transmission(self, samples: int, *, device: torch.device, real_dtype: torch.dtype) -> Tensor:
        """Return the total complex field transfer over one finite record."""
        bins = self.frequency_bin_indices(samples).to(device=device)
        t = torch.arange(samples, device=device, dtype=real_dtype) / self.grid.sample_rate_hz
        frequencies = self.modulation_frequencies_hz.to(device=device, dtype=real_dtype)
        indices = self.modulation_indices_rad.to(device=device, dtype=real_dtype)
        phases = self.rf_phases_rad.to(device=device, dtype=real_dtype)
        phase = (indices[:, None] * torch.sin(2.0 * torch.pi * frequencies[:, None] * t + phases[:, None])).sum(dim=0)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=device, dtype=real_dtype)
        amplitude_loss = torch.pow(torch.tensor(10.0, device=device, dtype=real_dtype), -loss_db / 20.0)
        return torch.polar(amplitude_loss.expand_as(phase), phase)

    def forward(self, field: Tensor) -> Tensor:
        """Phase-modulate a complex envelope along its final sample dimension."""
        require_complex_field(field)
        transfer = self.field_transmission(field.shape[-1], device=field.device, real_dtype=field.real.dtype)
        return field * transfer
