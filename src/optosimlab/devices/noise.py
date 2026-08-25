"""Loss and noise primitives for non-ideal optical system simulations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import nonnegative_parameter, require_complex_field


class OpticalAttenuator(nn.Module):
    """Power attenuator with a non-negative attenuation in dB."""

    def __init__(self, attenuation_db: float = 0.0, *, trainable: bool = False) -> None:
        super().__init__()
        if attenuation_db < 0:
            raise ValueError("attenuation_db must not be negative")
        self.attenuation_db = nn.Parameter(torch.tensor(float(attenuation_db)), requires_grad=trainable)

    def forward(self, field: Tensor) -> Tensor:
        require_complex_field(field)
        loss_db = nonnegative_parameter(self.attenuation_db, "attenuation_db").to(device=field.device, dtype=field.real.dtype)
        ten = torch.tensor(10.0, device=field.device, dtype=field.real.dtype)
        return field * torch.pow(ten, -loss_db / 20.0)


class AdditiveComplexGaussianNoise(nn.Module):
    """Add circular complex Gaussian field noise with RMS field value ``sigma``.

    Real and imaginary quadratures each have variance ``sigma**2 / 2``. Thus
    the expected total noise power is ``E[|n|^2] = sigma**2``.
    """

    def __init__(self, sigma: float = 0.0, *, trainable: bool = False) -> None:
        super().__init__()
        if sigma < 0:
            raise ValueError("sigma must not be negative")
        self.sigma = nn.Parameter(torch.tensor(float(sigma)), requires_grad=trainable)

    def forward(self, field: Tensor) -> Tensor:
        require_complex_field(field)
        sigma = nonnegative_parameter(self.sigma, "sigma").to(device=field.device, dtype=field.real.dtype)
        scale = sigma / torch.sqrt(torch.tensor(2.0, device=field.device, dtype=field.real.dtype))
        noise = torch.complex(torch.randn_like(field.real) * scale, torch.randn_like(field.real) * scale)
        return field + noise
