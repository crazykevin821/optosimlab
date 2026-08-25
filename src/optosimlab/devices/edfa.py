"""Deterministic small-signal erbium-doped fibre amplifier model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import C_LIGHT, H_PLANCK, SimulationGrid
from .base import nonnegative_parameter, require_complex_field


class SmallSignalEDFA(nn.Module):
    """Frequency-dependent, deterministic small-signal EDFA.

    The model deliberately covers the linear operating regime only.  Its peak
    *net* power gain in dB is

    ``G_peak_dB = gain_slope_dB_per_mA * max(I_pump - I_transparency, 0) - L``.

    Around ``gain_center_offset_hz`` the relative power-gain profile is a
    Gaussian with full width at half maximum ``gain_bandwidth_hz``:

    ``G_power(f) = 10**(G_peak_dB / 10) * exp[-4 ln(2) ((f-f0)/B)^2]``.

    Therefore the power gain at ``f0 +/- B/2`` is exactly half its peak
    (a -3.0103 dB relative drop).  The optical *field* receives
    ``sqrt(G_power)``.  This version has no ASE noise, gain saturation, pump
    dynamics, polarization dependence or transient population dynamics.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        pump_current_ma: float = 100.0,
        transparency_current_ma: float = 20.0,
        gain_slope_db_per_ma: float = 0.25,
        gain_bandwidth_hz: float = 4e12,
        gain_center_offset_hz: float = 0.0,
        insertion_loss_db: float = 0.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if pump_current_ma < 0:
            raise ValueError("pump_current_ma must not be negative")
        if transparency_current_ma < 0:
            raise ValueError("transparency_current_ma must not be negative")
        if gain_slope_db_per_ma < 0:
            raise ValueError("gain_slope_db_per_ma must not be negative")
        if gain_bandwidth_hz <= 0:
            raise ValueError("gain_bandwidth_hz must be positive")
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.grid = grid
        self.pump_current_ma = nn.Parameter(torch.tensor(float(pump_current_ma)), requires_grad=trainable)
        self.transparency_current_ma = nn.Parameter(torch.tensor(float(transparency_current_ma)), requires_grad=trainable)
        self.gain_slope_db_per_ma = nn.Parameter(torch.tensor(float(gain_slope_db_per_ma)), requires_grad=trainable)
        self.gain_bandwidth_hz = nn.Parameter(torch.tensor(float(gain_bandwidth_hz)), requires_grad=trainable)
        self.gain_center_offset_hz = nn.Parameter(torch.tensor(float(gain_center_offset_hz)), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def peak_net_gain_db(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Return current-mapped peak net power gain in dB."""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        pump = nonnegative_parameter(self.pump_current_ma, "pump_current_ma").to(device=device, dtype=dtype)
        transparency = nonnegative_parameter(self.transparency_current_ma, "transparency_current_ma").to(device=device, dtype=dtype)
        slope = nonnegative_parameter(self.gain_slope_db_per_ma, "gain_slope_db_per_ma").to(device=device, dtype=dtype)
        loss = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=device, dtype=dtype)
        return slope * (pump - transparency).clamp_min(0.0) - loss

    def power_gain(self, samples: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Return FFT-order power gain for a finite record.

        ``gain_bandwidth_hz`` is the FWHM of the relative *power* gain, not a
        field-amplitude width and not a bandwidth measured at an arbitrary dB
        level.
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        bandwidth = nonnegative_parameter(self.gain_bandwidth_hz, "gain_bandwidth_hz", floor=torch.finfo(dtype).eps).to(device=device, dtype=dtype)
        centre = self.gain_center_offset_hz.to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        relative_gain = torch.exp(-4.0 * math.log(2.0) * ((frequency - centre) / bandwidth).square())
        ten = torch.tensor(10.0, device=device, dtype=dtype)
        return torch.pow(ten, self.peak_net_gain_db(device=device, dtype=dtype) / 10.0) * relative_gain

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the real field-amplitude transfer function ``sqrt(G_power)``."""
        return torch.sqrt(self.power_gain(samples, device=device, dtype=dtype))

    def forward(self, field: Tensor) -> Tensor:
        """Apply deterministic frequency-dependent small-signal amplification."""
        require_complex_field(field)
        transfer = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.fft(field, dim=-1) * transfer, dim=-1)


class SaturatedNoisyEDFA(SmallSignalEDFA):
    """Small-signal EDFA extended with phenomenological saturation and ASE.

    For average input signal power ``P_in`` the deterministic gain is
    ``G_eff(f) = G_ss(f) / (1 + P_in / P_sat)``.  This is a scalar-gain
    saturation approximation: it preserves the spectral shape while reducing
    its level, and is appropriate for differentiable link-level experiments,
    not for modelling population dynamics.

    ASE follows a one-polarization complex-envelope convention.  With
    ``NF_linear = 10**(NF_dB / 10)`` and ``n_sp = NF_linear / 2``, its two-sided
    output PSD is ``S_ASE(f) = n_sp h nu max(G_eff(f)-1, 0)`` in W/Hz.
    For FFT size ``N`` and bin width ``df``, an unnormalised FFT noise bin has
    variance ``N**2 S_ASE df``.  PyTorch's IFFT then gives the correct expected
    time-domain mean noise power ``sum(S_ASE df)``.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        pump_current_ma: float = 100.0,
        transparency_current_ma: float = 20.0,
        gain_slope_db_per_ma: float = 0.25,
        gain_bandwidth_hz: float = 4e12,
        gain_center_offset_hz: float = 0.0,
        insertion_loss_db: float = 0.0,
        saturation_power: float = 1.0,
        noise_figure_db: float = 5.0,
        *,
        enable_ase: bool = True,
        trainable: bool = True,
    ) -> None:
        if saturation_power <= 0:
            raise ValueError("saturation_power must be positive")
        if noise_figure_db < 0:
            raise ValueError("noise_figure_db must not be negative")
        super().__init__(
            grid,
            pump_current_ma=pump_current_ma,
            transparency_current_ma=transparency_current_ma,
            gain_slope_db_per_ma=gain_slope_db_per_ma,
            gain_bandwidth_hz=gain_bandwidth_hz,
            gain_center_offset_hz=gain_center_offset_hz,
            insertion_loss_db=insertion_loss_db,
            trainable=trainable,
        )
        self.saturation_power = nn.Parameter(torch.tensor(float(saturation_power)), requires_grad=trainable)
        self.noise_figure_db = nn.Parameter(torch.tensor(float(noise_figure_db)), requires_grad=trainable)
        self.enable_ase = bool(enable_ase)

    def effective_power_gain(
        self,
        samples: int,
        signal_power: Tensor | float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return saturated power gain; leading dimensions follow ``signal_power``."""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        power = torch.as_tensor(signal_power, device=device, dtype=dtype)
        saturation = nonnegative_parameter(self.saturation_power, "saturation_power", floor=torch.finfo(dtype).eps).to(device=device, dtype=dtype)
        return self.power_gain(samples, device=device, dtype=dtype) / (1.0 + power / saturation)

    def ase_power_spectral_density(
        self,
        samples: int,
        signal_power: Tensor | float = 0.0,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return one-polarization complex-envelope ASE PSD in W/Hz."""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        gain = self.effective_power_gain(samples, signal_power, device=device, dtype=dtype)
        noise_figure = nonnegative_parameter(self.noise_figure_db, "noise_figure_db").to(device=device, dtype=dtype)
        n_sp = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), noise_figure / 10.0) / 2.0
        photon_energy = torch.tensor(H_PLANCK * C_LIGHT / self.grid.center_wavelength_m, device=device, dtype=dtype)
        return n_sp * photon_energy * (gain - 1.0).clamp_min(0.0)

    def forward(self, field: Tensor) -> Tensor:
        """Amplify ``field`` with saturation and, optionally, statistically correct ASE."""
        require_complex_field(field)
        samples = field.shape[-1]
        signal_power = field.abs().square().mean(dim=-1, keepdim=True)
        gain = self.effective_power_gain(samples, signal_power, device=field.device, dtype=field.real.dtype)
        signal = torch.fft.ifft(torch.fft.fft(field, dim=-1) * torch.sqrt(gain), dim=-1)
        if not self.enable_ase:
            return signal

        psd = self.ase_power_spectral_density(samples, signal_power, device=field.device, dtype=field.real.dtype)
        bin_width_hz = self.grid.sample_rate_hz / samples
        bin_variance = samples**2 * psd * bin_width_hz
        quadrature_std = torch.sqrt(bin_variance / 2.0)
        spectral_noise = torch.complex(
            torch.randn_like(field.real) * quadrature_std,
            torch.randn_like(field.real) * quadrature_std,
        )
        return signal + torch.fft.ifft(spectral_noise, dim=-1)
