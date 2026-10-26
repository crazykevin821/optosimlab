"""含 RF 一阶响应与相位扩散的实际电光梳。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field
from .electro_optic_comb import ElectroOpticComb


class PracticalElectroOpticComb(ElectroOpticComb):
    """含因果 RF 带宽与 Wiener RF 相位噪声的电光梳。

    每一级采用一阶电响应 ``H_rf(f)=1/(1+jf/f_c)``，因此理想调制指数乘以
    ``|H_rf|``，RF 相位附加 ``arg(H_rf)=-atan(f/f_c)``。可选相噪使用
    Wiener 随机游走：``phi_noise[n+1]-phi_noise[n] ~ N(0,D_phi*dt)``。
    ``D_phi`` 的单位为 rad^2/s 且可训练。该模型提供量纲明确、可控制的
    扩散近似，不宣称复现 RF 振荡器的完整相噪谱模板。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        modulation_frequency_hz: float | Tensor,
        modulation_indices_rad: float | Tensor = 1.0,
        rf_phases_rad: float | Tensor = 0.0,
        insertion_loss_db: float = 0.0,
        rf_bandwidth_hz: float | Tensor = 1e12,
        rf_phase_noise_diffusion_rad2_per_s: float | Tensor = 0.0,
        *,
        enable_phase_noise: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__(
            grid,
            modulation_frequency_hz=modulation_frequency_hz,
            modulation_indices_rad=modulation_indices_rad,
            rf_phases_rad=rf_phases_rad,
            insertion_loss_db=insertion_loss_db,
            trainable=trainable,
        )
        bandwidth = self._stage_vector(rf_bandwidth_hz, "rf_bandwidth_hz")
        if torch.any(bandwidth <= 0):
            raise ValueError("rf_bandwidth_hz must be positive")
        diffusion = self._stage_vector(rf_phase_noise_diffusion_rad2_per_s, "rf_phase_noise_diffusion_rad2_per_s")
        if torch.any(diffusion < 0):
            raise ValueError("rf_phase_noise_diffusion_rad2_per_s must not be negative")
        self.rf_bandwidth_hz = nn.Parameter(bandwidth, requires_grad=trainable)
        self.rf_phase_noise_diffusion_rad2_per_s = nn.Parameter(diffusion, requires_grad=trainable)
        self.enable_phase_noise = bool(enable_phase_noise)

    def _stage_vector(self, value: float | Tensor, name: str) -> Tensor:
        vector = torch.as_tensor(value, dtype=torch.float32)
        if vector.ndim == 0:
            vector = vector.expand(self.stage_count).clone()
        if vector.shape != self.modulation_indices_rad.shape:
            raise ValueError(f"{name} must be a scalar or have one entry per modulation stage")
        return vector

    def rf_amplitude_and_phase(self, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """返回每一级 RF 一阶幅度和相位响应。"""
        bandwidth = nonnegative_parameter(self.rf_bandwidth_hz, "rf_bandwidth_hz", floor=torch.finfo(dtype).eps).to(device=device, dtype=dtype)
        frequency = self.modulation_frequencies_hz.to(device=device, dtype=dtype)
        ratio = frequency / bandwidth
        return torch.rsqrt(1.0 + ratio.square()), -torch.atan(ratio)

    def phase_noise(self, samples: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> Tensor:
        """采样各级 Wiener 相位轨迹，并令初始相位为零。"""
        if not self.enable_phase_noise:
            return torch.zeros((self.stage_count, samples), device=device, dtype=dtype)
        diffusion = nonnegative_parameter(
            self.rf_phase_noise_diffusion_rad2_per_s,
            "rf_phase_noise_diffusion_rad2_per_s",
        ).to(device=device, dtype=dtype)
        innovations = torch.randn((self.stage_count, samples), device=device, dtype=dtype, generator=generator)
        innovations[:, 0] = 0.0
        return torch.cumsum(innovations * torch.sqrt(diffusion[:, None] * self.grid.dt_s), dim=-1)

    def field_transmission(
        self,
        samples: int,
        *,
        device: torch.device,
        real_dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """返回包含 RF 带宽和相位扩散的场传输。"""
        self.frequency_bin_indices(samples)
        t = torch.arange(samples, device=device, dtype=real_dtype) / self.grid.sample_rate_hz
        frequencies = self.modulation_frequencies_hz.to(device=device, dtype=real_dtype)
        indices = self.modulation_indices_rad.to(device=device, dtype=real_dtype)
        phases = self.rf_phases_rad.to(device=device, dtype=real_dtype)
        amplitude, lag = self.rf_amplitude_and_phase(device=device, dtype=real_dtype)
        stochastic_phase = self.phase_noise(samples, device=device, dtype=real_dtype, generator=generator)
        phase = (
            indices[:, None]
            * amplitude[:, None]
            * torch.sin(2.0 * torch.pi * frequencies[:, None] * t + phases[:, None] + lag[:, None] + stochastic_phase)
        ).sum(dim=0)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=device, dtype=real_dtype)
        amplitude_loss = torch.pow(torch.tensor(10.0, device=device, dtype=real_dtype), -loss_db / 20.0)
        return torch.polar(amplitude_loss.expand_as(phase), phase)

    def forward(self, field: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """对 ``field`` 施加受 RF 带宽限制且可选带相噪的电光调制。"""
        require_complex_field(field)
        transfer = self.field_transmission(
            field.shape[-1],
            device=field.device,
            real_dtype=field.real.dtype,
            generator=generator,
        )
        return field * transfer
