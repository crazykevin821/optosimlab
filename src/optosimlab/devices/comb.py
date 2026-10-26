"""定义在 FFT 采样网格上的确定性光学频率梳。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import GraphDevice, nonnegative_parameter


class OpticalFrequencyComb(GraphDevice):
    graph_input_ports = ("samples",)
    """生成以载波为中心的离散光学频率梳。

    梳齿位于 ``line_spacing_hz`` 的整数倍。奇数梳齿关于载波对称；偶数梳齿
    使用 ``-line_count/2,...,line_count/2-1``，仍包含载波。每个梳齿具有
    可训练功率和相位。给定记录长度时梳齿必须严格落在 FFT 频点上；本确定性
    模型会拒绝未对齐间隔，避免悄然引入频谱泄漏。

    若 ``P_k`` 为梳齿功率、``phi_k`` 为相位，则未归一化 DFT 系数为
    ``N*sqrt(P_k)*exp(j*phi_k)``。对不同频点使用 PyTorch 逆 FFT 后，
    波形平均功率严格等于 ``sum(P_k)``。所有频率均为相对光载波的频偏。
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
        if line_count < 1:
            raise ValueError("line_count must be a positive integer")
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
        """以场平方单位返回各梳齿配置功率之和。"""
        return nonnegative_parameter(self.line_powers, "line_powers").sum()

    def line_bin_indices(self, samples: int) -> Tensor:
        """校验频点对齐和带宽后返回 FFT 原生索引。"""
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
        offsets = torch.arange(-(self.line_count // 2), self.line_count - self.line_count // 2, dtype=torch.long)
        signed_bins = offsets * integer_spacing
        if int(signed_bins.abs().max()) > (samples - 1) // 2:
            raise ValueError("comb lines exceed the representable positive/negative frequency range")
        return signed_bins.remainder(samples)

    def line_offsets_hz(self, samples: int, *, device: torch.device | None = None) -> Tensor:
        """返回与功率和相位参数同序的载频偏移。"""
        indices = self.line_bin_indices(samples).to(device=device)
        signed = torch.where(indices <= samples // 2, indices, indices - samples)
        return signed.to(torch.float64) * (self.grid.sample_rate_hz / samples)

    def nominal_line_offsets_hz(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """不检查有限记录频点对齐，直接返回配置的载频偏移。"""
        offsets = torch.arange(
            -(self.line_count // 2),
            self.line_count - self.line_count // 2,
            device=device,
            dtype=dtype,
        )
        offsets = offsets * self.line_spacing_hz
        if torch.any(offsets.abs() > self.grid.sample_rate_hz / 2.0):
            raise ValueError("comb lines exceed the representable positive/negative frequency range")
        return offsets

    def spectrum(self, samples: int, *, device: torch.device | None = None, real_dtype: torch.dtype | None = None) -> Tensor:
        """返回光梳波形未经归一化的 FFT 频谱。"""
        indices = self.line_bin_indices(samples)
        device = device or self.line_powers.device
        dtype = real_dtype or self.line_powers.dtype
        powers = nonnegative_parameter(self.line_powers, "line_powers").to(device=device, dtype=dtype)
        phases = self.line_phases_rad.to(device=device, dtype=dtype)
        coefficients = torch.polar(samples * torch.sqrt(powers), phases)
        base = torch.zeros(samples, device=device, dtype=coefficients.dtype)
        return base.scatter(0, indices.to(device=device), coefficients)

    def forward(self, samples: int, *, device: torch.device | None = None, real_dtype: torch.dtype | None = None) -> Tensor:
        """生成包含 ``samples`` 点的一维复包络。"""
        return torch.fft.ifft(self.spectrum(samples, device=device, real_dtype=real_dtype))

    def channel_fields(
        self,
        samples: int,
        *,
        device: torch.device | None = None,
        real_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """返回形状为 ``(line_count, samples)`` 的分离梳齿包络。

        每一行是单位为 sqrt(W) 的一路相干音调，其时间平均功率严格等于
        对应的 ``line_powers``。与 ``spectrum()`` 不同，此分通道表示不要求
        有限记录包含整数个音调周期。
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        device = device or self.line_powers.device
        dtype = real_dtype or self.line_powers.dtype
        powers = nonnegative_parameter(self.line_powers, "line_powers").to(device=device, dtype=dtype)
        phases = self.line_phases_rad.to(device=device, dtype=dtype)
        offsets = torch.arange(
            -(self.line_count // 2), self.line_count - self.line_count // 2,
            device=device, dtype=dtype,
        ) * self.line_spacing_hz
        if torch.any(offsets.abs() > self.grid.sample_rate_hz / 2):
            raise ValueError("comb lines exceed the representable positive/negative frequency range")
        time = torch.arange(samples, device=device, dtype=dtype) / self.grid.sample_rate_hz
        angles = 2.0 * torch.pi * offsets.unsqueeze(-1) * time + phases.unsqueeze(-1)
        return torch.sqrt(powers).unsqueeze(-1) * torch.polar(torch.ones_like(angles), angles)

    def channel_envelopes(
        self,
        samples: int,
        *,
        device: torch.device | None = None,
        real_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """为每个光梳载波返回一路基带包络。

        载频由 ``nominal_line_offsets_hz()`` 另行提供。该表示用于显式使用
        每个通道载频偏移的 WDM 器件。
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        device = device or self.line_powers.device
        dtype = real_dtype or self.line_powers.dtype
        self.nominal_line_offsets_hz(device=device, dtype=dtype)
        powers = nonnegative_parameter(self.line_powers, "line_powers").to(device=device, dtype=dtype)
        phases = self.line_phases_rad.to(device=device, dtype=dtype)
        coefficients = torch.polar(torch.sqrt(powers), phases)
        return coefficients.unsqueeze(-1).expand(self.line_count, samples)
