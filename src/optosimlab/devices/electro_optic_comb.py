"""时域电光频率梳模型。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import GraphDevice, nonnegative_parameter, require_complex_field


class ElectroOpticComb(GraphDevice):
    """级联由相干 RF 音调驱动的理想相位调制器。

    每一级施加无损相位调制传递函数

    ``E_out(t) = E_in(t) exp(j * beta * sin(2 pi f_rf t + phi_rf))``.

    对连续波输入，Jacobi--Anger 展开会在 ``n*f_rf`` 频偏处产生复幅为
    ``J_n(beta)*exp(j*n*phi_rf)`` 的梳齿。每一级传递函数的模为一，故级联
    不改变可选总插损之前的光功率。RF 频率刻意作为结构值而不是参数保存：
    它必须落在整数 FFT 频点上，有限记录才能无泄漏地表示周期光梳。
    调制指数、RF 相位和插损均注册为可优化的 ``nn.Parameter``。
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
        """级联 RF 相位调制级数。"""
        return int(self.modulation_indices_rad.numel())

    def frequency_bin_indices(self, samples: int) -> Tensor:
        """校验 RF 音调与记录周期对齐，并返回其正频 FFT 索引。"""
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
        """返回一个有限记录内的总复场传输。"""
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
        """沿最后一个样本维对复包络进行相位调制。"""
        require_complex_field(field)
        transfer = self.field_transmission(field.shape[-1], device=field.device, real_dtype=field.real.dtype)
        return field * transfer
