"""Physical-unit conventions and conversion helpers.

OptoSimLab uses SI units at public interfaces.  A complex optical envelope E
is normalized so that abs(E)**2 is instantaneous optical power in watts.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

DEFAULT_UNITS = {
    "optical_field": "sqrt(W)",
    "optical_power": "W",
    "display_power": "dBm",
    "voltage": "V",
    "time": "s",
    "frequency": "Hz",
    "wavelength": "m",
    "length": "m",
    "phase": "rad",
    "gain_loss": "dB (power ratio)",
    "dispersion": "s^2/m",
}


def mw_to_w(power_mw: Tensor | float) -> Tensor:
    """Convert milliwatts to watts."""
    return torch.as_tensor(power_mw) * 1e-3


def w_to_mw(power_w: Tensor | float) -> Tensor:
    """Convert watts to milliwatts."""
    return torch.as_tensor(power_w) * 1e3


def dbm_to_w(power_dbm: Tensor | float) -> Tensor:
    """Convert dBm to watts."""
    value = torch.as_tensor(power_dbm)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 10.0) * 1e-3


def w_to_dbm(power_w: Tensor | float) -> Tensor:
    """Convert strictly positive power in watts to dBm."""
    value = torch.as_tensor(power_w)
    if torch.any(value <= 0):
        raise ValueError("power_w must be positive")
    return 10.0 * torch.log10(value / 1e-3)


def db_to_power_ratio(value_db: Tensor | float) -> Tensor:
    """Convert a power gain/loss in dB to a dimensionless power ratio."""
    value = torch.as_tensor(value_db)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 10.0)


def db_to_field_ratio(value_db: Tensor | float) -> Tensor:
    """Convert a power gain/loss in dB to the corresponding field ratio."""
    value = torch.as_tensor(value_db)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 20.0)


def power_ratio_to_db(ratio: Tensor | float) -> Tensor:
    """Convert a strictly positive power ratio to dB."""
    value = torch.as_tensor(ratio)
    if torch.any(value <= 0):
        raise ValueError("ratio must be positive")
    return 10.0 * torch.log10(value)


def field_from_power_w(power_w: Tensor | float, phase_rad: Tensor | float = 0.0) -> Tensor:
    """Create a complex field in sqrt(W) from power in W and phase in rad."""
    power = torch.as_tensor(power_w)
    if not power.is_floating_point():
        power = power.to(torch.get_default_dtype())
    if torch.any(power < 0):
        raise ValueError("power_w must not be negative")
    phase = torch.as_tensor(phase_rad, device=power.device, dtype=power.dtype)
    return torch.polar(torch.sqrt(power), phase)


def field_power_w(field: Tensor) -> Tensor:
    """Return instantaneous optical power in W from a complex field."""
    if not torch.is_complex(field):
        raise TypeError("field must be a complex tensor")
    return field.abs().square()


DBM_REFERENCE_W = 1e-3
SQRT_W_UNIT = math.sqrt(1.0)
