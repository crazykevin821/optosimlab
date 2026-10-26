"""物理单位约定与换算工具。

OptoSimLab 的公开接口统一使用 SI 单位。复光包络 ``E`` 的归一化约定为
``abs(E)**2`` 等于瞬时光功率，单位为瓦。
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
    """把毫瓦换算为瓦。"""
    return torch.as_tensor(power_mw) * 1e-3


def w_to_mw(power_w: Tensor | float) -> Tensor:
    """把瓦换算为毫瓦。"""
    return torch.as_tensor(power_w) * 1e3


def dbm_to_w(power_dbm: Tensor | float) -> Tensor:
    """把 dBm 换算为瓦。"""
    value = torch.as_tensor(power_dbm)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 10.0) * 1e-3


def w_to_dbm(power_w: Tensor | float) -> Tensor:
    """把严格为正的瓦功率换算为 dBm。"""
    value = torch.as_tensor(power_w)
    if torch.any(value <= 0):
        raise ValueError("power_w must be positive")
    return 10.0 * torch.log10(value / 1e-3)


def db_to_power_ratio(value_db: Tensor | float) -> Tensor:
    """把功率增益或损耗 dB 换算为无量纲功率比。"""
    value = torch.as_tensor(value_db)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 10.0)


def db_to_field_ratio(value_db: Tensor | float) -> Tensor:
    """把功率增益或损耗 dB 换算为对应场幅度比。"""
    value = torch.as_tensor(value_db)
    return torch.pow(torch.as_tensor(10.0, device=value.device, dtype=value.dtype), value / 20.0)


def power_ratio_to_db(ratio: Tensor | float) -> Tensor:
    """把严格为正的功率比换算为 dB。"""
    value = torch.as_tensor(ratio)
    if torch.any(value <= 0):
        raise ValueError("ratio must be positive")
    return 10.0 * torch.log10(value)


def field_from_power_w(power_w: Tensor | float, phase_rad: Tensor | float = 0.0) -> Tensor:
    """根据功率 W 和相位 rad 生成单位为 sqrt(W) 的复光场。"""
    power = torch.as_tensor(power_w)
    if not power.is_floating_point():
        power = power.to(torch.get_default_dtype())
    if torch.any(power < 0):
        raise ValueError("power_w must not be negative")
    phase = torch.as_tensor(phase_rad, device=power.device, dtype=power.dtype)
    return torch.polar(torch.sqrt(power), phase)


def field_power_w(field: Tensor) -> Tensor:
    """根据复光场返回瞬时光功率 W。"""
    if not torch.is_complex(field):
        raise TypeError("field must be a complex tensor")
    return field.abs().square()


DBM_REFERENCE_W = 1e-3
SQRT_W_UNIT = math.sqrt(1.0)
