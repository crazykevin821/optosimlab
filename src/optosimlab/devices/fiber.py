"""Linear dispersive single-mode fibre model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import C_LIGHT, SimulationGrid
from .base import GraphDevice, nonnegative_parameter, require_complex_field


class LinearDispersiveFiber(GraphDevice):
    """Frequency-domain fibre with power attenuation and second-order dispersion.

    In this package, forward FFT uses ``exp(-j 2 pi f t)`` and inverse FFT
    uses ``exp(+j 2 pi f t)``. After removing beta0 and beta1, propagation is
    therefore ``H(f) = exp(-alpha_power L / 2) exp(+j beta2 (2 pi f)^2 L / 2)``.
    ``alpha_power`` is in nepers/m and ``beta2`` is in s^2/m.  The attenuation
    argument is specified in dB/km as a *power* loss, not field loss.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        length_m: float,
        *,
        attenuation_db_per_km: float = 0.0,
        beta2_s2_per_m: float | None = None,
        dispersion_ps_nm_km: float | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if length_m < 0:
            raise ValueError("length_m must not be negative")
        if attenuation_db_per_km < 0:
            raise ValueError("attenuation_db_per_km must not be negative")
        if beta2_s2_per_m is not None and dispersion_ps_nm_km is not None:
            raise ValueError("provide beta2_s2_per_m or dispersion_ps_nm_km, not both")
        if dispersion_ps_nm_km is not None:
            beta2_s2_per_m = self.beta2_from_dispersion(dispersion_ps_nm_km, grid.center_wavelength_m)
        self.grid = grid
        self.length_m = nn.Parameter(torch.tensor(float(length_m)), requires_grad=trainable)
        self.attenuation_db_per_km = nn.Parameter(torch.tensor(float(attenuation_db_per_km)), requires_grad=trainable)
        self.beta2_s2_per_m = nn.Parameter(torch.tensor(float(beta2_s2_per_m or 0.0)), requires_grad=trainable)

    @staticmethod
    def beta2_from_dispersion(dispersion_ps_nm_km: float, wavelength_m: float) -> float:
        """Convert D [ps/(nm km)] to beta2 [s^2/m]."""
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        d_s_per_m2 = dispersion_ps_nm_km * 1e-6
        return -d_s_per_m2 * wavelength_m**2 / (2.0 * math.pi * C_LIGHT)

    @staticmethod
    def dispersion_from_beta2(beta2_s2_per_m: float, wavelength_m: float) -> float:
        """Convert beta2 [s^2/m] to D [ps/(nm km)]."""
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        return -2.0 * math.pi * C_LIGHT * beta2_s2_per_m / wavelength_m**2 / 1e-6

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        length = nonnegative_parameter(self.length_m, "length_m").to(device=device, dtype=dtype)
        attenuation_db = nonnegative_parameter(self.attenuation_db_per_km, "attenuation_db_per_km").to(device=device, dtype=dtype)
        beta2 = self.beta2_s2_per_m.to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        omega = 2.0 * torch.pi * frequency
        # dB/km -> Np/m for power. Field amplitude receives half the exponent.
        alpha_power_np_per_m = attenuation_db * (math.log(10.0) / 10.0) / 1000.0
        log_amplitude = -0.5 * alpha_power_np_per_m * length
        phase = 0.5 * beta2 * omega.square() * length
        return torch.exp(torch.complex(log_amplitude.expand_as(phase), phase))

    def wdm_transfer_function(
        self,
        samples: int,
        carrier_offsets_hz: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        reference_offset_hz: Tensor | float | None = None,
    ) -> Tensor:
        """Return narrowband WDM transfer with wavelength-dependent delay.

        Expanding the fibre phase around carrier offset f_k gives the baseband
        linear term beta2*L*(2*pi*f_k)*(2*pi*f).  A common group delay is
        removed by referencing the first (or supplied) carrier.  The resulting
        adjacent-channel delay is -beta2*L*2*pi*Delta_f under this package's
        FFT sign convention.  This method intentionally omits the common
        intra-channel quadratic term so it models the progressive-delay regime
        used by transversal photonic processors.
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not isinstance(carrier_offsets_hz, Tensor) or carrier_offsets_hz.ndim != 1:
            raise ValueError("carrier_offsets_hz must be a one-dimensional tensor")
        offsets = carrier_offsets_hz.to(device=device, dtype=dtype)
        reference = offsets[0] if reference_offset_hz is None else torch.as_tensor(reference_offset_hz, device=device, dtype=dtype)
        relative_carrier_omega = 2.0 * torch.pi * (offsets - reference)
        baseband_omega = 2.0 * torch.pi * self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        length = nonnegative_parameter(self.length_m, "length_m").to(device=device, dtype=dtype)
        attenuation_db = nonnegative_parameter(self.attenuation_db_per_km, "attenuation_db_per_km").to(
            device=device, dtype=dtype
        )
        beta2 = self.beta2_s2_per_m.to(device=device, dtype=dtype)
        alpha_power_np_per_m = attenuation_db * (math.log(10.0) / 10.0) / 1000.0
        log_amplitude = -0.5 * alpha_power_np_per_m * length
        phase = beta2 * length * relative_carrier_omega.unsqueeze(-1) * baseband_omega.unsqueeze(0)
        return torch.exp(torch.complex(log_amplitude.expand_as(phase), phase))

    def adjacent_channel_delay_s(self, line_spacing_hz: float) -> Tensor:
        """Return group-delay increment between increasing-frequency channels."""
        if line_spacing_hz <= 0:
            raise ValueError("line_spacing_hz must be positive")
        return -self.beta2_s2_per_m * self.length_m * (2.0 * math.pi * line_spacing_hz)

    def forward(self, field: Tensor, carrier_offsets_hz: Tensor | None = None) -> Tensor:
        require_complex_field(field)
        if carrier_offsets_hz is not None:
            if field.ndim < 2 or field.shape[-2] != carrier_offsets_hz.numel():
                raise ValueError("WDM field must have shape (..., channels, samples) matching carrier_offsets_hz")
            response = self.wdm_transfer_function(
                field.shape[-1], carrier_offsets_hz, device=field.device, dtype=field.real.dtype
            )
            while response.ndim < field.ndim:
                response = response.unsqueeze(0)
            return torch.fft.ifft(torch.fft.fft(field, dim=-1) * response, dim=-1)
        response = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.fft(field, dim=-1) * response, dim=-1)
