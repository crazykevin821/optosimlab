"""Simulation-wide physical constants and sampling-grid helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

C_LIGHT = 299_792_458.0  # vacuum light speed, m/s
H_PLANCK = 6.626_070_15e-34  # Planck constant, J s (exact SI value)


@dataclass(frozen=True)
class SimulationGrid:
    """Sampling contract shared by frequency-domain devices.

    The represented signal is a complex optical envelope.  The frequency grid
    therefore contains offsets from the optical carrier, not the carrier itself.
    """

    sample_rate_hz: float
    center_wavelength_m: float = 1550e-9

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.center_wavelength_m <= 0:
            raise ValueError("center_wavelength_m must be positive")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    def frequencies_hz(self, samples: int, *, device: torch.device | None = None) -> torch.Tensor:
        """Return FFT-order frequencies (zero, positive, then negative bins)."""
        if samples <= 0:
            raise ValueError("samples must be positive")
        return torch.fft.fftfreq(samples, d=self.dt_s, device=device)
