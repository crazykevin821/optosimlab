"""可微分、可编程的频域整形器。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import GraphDevice, nonnegative_parameter, require_complex_field


class WaveShaper(GraphDevice):
    """可训练的 WaveShaper 类衰减与相位掩模。

    控制点按 fftshift 后的频率轴从负频偏到正频偏排列，中间控制点代表
    光载波。响应会线性插值到当前 FFT 网格。``attenuation_db`` 受非负
    约束以符合无源 WaveShaper，``phase_rad`` 则允许任意相位。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        control_points: int = 129,
        *,
        control_frequencies_hz: Tensor | None = None,
        attenuation_db: Tensor | None = None,
        phase_rad: Tensor | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if control_frequencies_hz is None:
            if control_points < 2:
                raise ValueError("control_points must be at least 2")
            frequencies = torch.linspace(-grid.sample_rate_hz / 2.0, grid.sample_rate_hz / 2.0, control_points)
        else:
            frequencies = torch.as_tensor(control_frequencies_hz, dtype=torch.float32)
            if frequencies.ndim != 1:
                raise ValueError("control_frequencies_hz must be one-dimensional")
            control_points = frequencies.numel()
        if control_points < 2:
            raise ValueError("control_points must be at least 2")
        if torch.any(torch.diff(frequencies) <= 0):
            raise ValueError("control_frequencies_hz must be strictly increasing")
        if torch.any(frequencies.abs() > grid.sample_rate_hz / 2.0):
            raise ValueError("control frequencies must lie inside the simulated Nyquist band")
        self.grid = grid
        self.control_points = int(control_points)
        self.register_buffer("control_frequencies_hz", frequencies)
        attenuation = torch.zeros(control_points) if attenuation_db is None else torch.as_tensor(attenuation_db, dtype=torch.float32)
        phase = torch.zeros(control_points) if phase_rad is None else torch.as_tensor(phase_rad, dtype=torch.float32)
        if attenuation.shape != (control_points,) or phase.shape != (control_points,):
            raise ValueError("attenuation_db and phase_rad must match the number of control frequencies")
        if torch.any(attenuation < 0):
            raise ValueError("attenuation_db must not be negative")
        self.attenuation_db = nn.Parameter(attenuation.clone(), requires_grad=trainable)
        self.phase_rad = nn.Parameter(phase.clone(), requires_grad=trainable)

    @staticmethod
    def _interpolate(query: Tensor, points: Tensor, values: Tensor) -> Tensor:
        query = query.clamp(min=points[0], max=points[-1])
        upper = torch.searchsorted(points, query, right=True).clamp(1, points.numel() - 1)
        lower = upper - 1
        fraction = (query - points[lower]) / (points[upper] - points[lower])
        return values[lower] + fraction * (values[upper] - values[lower])

    def transfer_at_frequencies(self, frequencies_hz: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        """返回任意载频偏移处的复场透射系数。"""
        if not isinstance(frequencies_hz, Tensor) or torch.is_complex(frequencies_hz):
            raise TypeError("frequencies_hz must be a real-valued torch.Tensor")
        dtype = dtype or (frequencies_hz.dtype if frequencies_hz.is_floating_point() else torch.float32)
        device = frequencies_hz.device
        points = self.control_frequencies_hz.to(device=device, dtype=dtype)
        attenuation = nonnegative_parameter(self.attenuation_db, "attenuation_db").to(device=device, dtype=dtype)
        phase = self.phase_rad.to(device=device, dtype=dtype)
        attenuation = self._interpolate(frequencies_hz.to(dtype=dtype), points, attenuation)
        phase = self._interpolate(frequencies_hz.to(dtype=dtype), points, phase)
        ten = torch.tensor(10.0, device=device, dtype=dtype)
        return torch.polar(torch.pow(ten, -attenuation / 20.0), phase)

    def power_transmission_at(self, frequencies_hz: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        return self.transfer_at_frequencies(frequencies_hz, dtype=dtype).abs().square()

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if samples <= 0:
            raise ValueError("samples must be positive")
        frequencies = torch.fft.fftshift(self.grid.frequencies_hz(samples, device=device)).to(dtype=dtype)
        return self.transfer_at_frequencies(frequencies, dtype=dtype)

    def forward(self, field: Tensor, channel_offsets_hz: Tensor | None = None) -> Tensor:
        require_complex_field(field)
        if channel_offsets_hz is not None:
            if field.ndim < 2 or channel_offsets_hz.ndim != 1 or field.shape[-2] != channel_offsets_hz.numel():
                raise ValueError("WDM field must have shape (..., channels, samples) matching channel_offsets_hz")
            response = self.transfer_at_frequencies(
                channel_offsets_hz.to(device=field.device), dtype=field.real.dtype
            )
            shape = [1] * field.ndim
            shape[-2] = response.numel()
            return field * response.view(shape)
        spectrum_shifted = torch.fft.fftshift(torch.fft.fft(field, dim=-1), dim=-1)
        response = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.ifftshift(spectrum_shifted * response, dim=-1), dim=-1)
