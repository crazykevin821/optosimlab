"""全局物理常数与采样网格工具。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

C_LIGHT = 299_792_458.0  # 真空光速，m/s
H_PLANCK = 6.626_070_15e-34  # 普朗克常数，J s（SI 精确定义值）


@dataclass(frozen=True)
class SimulationGrid:
    """频域器件共享的采样约定。

    仿真信号采用复光包络表示，因此频率网格给出相对光载波的频偏，
    而不是数百 THz 的绝对光载频。
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
        """返回 FFT 原生顺序的频率：零频、正频、负频。"""
        if samples <= 0:
            raise ValueError("samples must be positive")
        return torch.fft.fftfreq(samples, d=self.dt_s, device=device)
