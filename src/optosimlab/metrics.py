"""验证信号级光学仿真的系统指标。"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import SimulationGrid
from .devices.base import require_complex_field


def mean_power(field: Tensor) -> Tensor:
    """以场平方单位返回平均光功率。"""
    require_complex_field(field)
    return field.abs().square().mean()


def normalized_mean_square_error(reference: Tensor, estimate: Tensor, *, eps: float = 1e-12) -> Tensor:
    """返回 NMSE：估计误差均方除以参考信号均方。"""
    require_complex_field(reference)
    require_complex_field(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    numerator = (estimate - reference).abs().square().mean()
    denominator = reference.abs().square().mean().clamp_min(eps)
    return numerator / denominator


def snr_db(signal: Tensor, noise: Tensor, *, eps: float = 1e-20) -> Tensor:
    """根据独立的复信号与噪声张量计算功率信噪比 dB。"""
    require_complex_field(signal)
    require_complex_field(noise)
    if signal.shape != noise.shape:
        raise ValueError("signal and noise must have the same shape")
    ratio = signal.abs().square().mean() / noise.abs().square().mean().clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def error_vector_magnitude(reference: Tensor, estimate: Tensor, *, eps: float = 1e-20) -> Tensor:
    """返回两个复包络之间无量纲的 RMS EVM。"""
    require_complex_field(reference)
    require_complex_field(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    return torch.sqrt((estimate - reference).abs().square().mean() / reference.abs().square().mean().clamp_min(eps))


def power_spectrum(field: Tensor) -> Tensor:
    """返回 fftshift 后逐频点功率，其总和等于平均功率。

    PyTorch 使用未归一化 FFT，因此每个输出频点为 ``|X[k]|^2/N^2``。
    依据帕塞瓦尔定理，沿末维求和等于 ``mean(|field|^2)``，并保留前导批次维。
    """
    require_complex_field(field)
    samples = field.shape[-1]
    spectrum = torch.fft.fftshift(torch.fft.fft(field, dim=-1), dim=-1)
    return spectrum.abs().square() / samples**2


def centered_frequency_axis_hz(grid: SimulationGrid, samples: int, *, device: torch.device | None = None) -> Tensor:
    """返回与 :func:`power_spectrum` 对齐的频率轴。"""
    return torch.fft.fftshift(grid.frequencies_hz(samples, device=device))


def eye_diagram(signal: Tensor, samples_per_symbol: int, *, offset_samples: int = 0) -> Tensor:
    """把信号折叠为相邻两个符号长度的眼图轨迹。

    返回形状为 ``signal.shape[:-1] + (轨迹数, 2*samples_per_symbol)``。
    相邻轨迹相隔一个符号开始，因此除记录边界外，每个 UI 会参与两条相邻眼图轨迹。
    """
    if not isinstance(signal, Tensor) or signal.ndim < 1:
        raise TypeError("signal must be a torch.Tensor with a final sample dimension")
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if offset_samples < 0:
        raise ValueError("offset_samples must not be negative")
    width = 2 * samples_per_symbol
    available = signal.shape[-1] - offset_samples
    if available < width:
        raise ValueError("signal does not contain one complete two-symbol eye trace after offset_samples")
    starts = torch.arange(0, available - width + 1, samples_per_symbol, device=signal.device)
    indices = offset_samples + starts[:, None] + torch.arange(width, device=signal.device)
    return signal[..., indices]


def gaussian_q_factor(level_zero: Tensor, level_one: Tensor, *, eps: float = 1e-20) -> Tensor:
    """根据两类实数样本返回二元高斯判决 Q 因子。"""
    if not isinstance(level_zero, Tensor) or not isinstance(level_one, Tensor):
        raise TypeError("level_zero and level_one must be torch.Tensor instances")
    if torch.is_complex(level_zero) or torch.is_complex(level_one):
        raise TypeError("Gaussian BER estimation requires real-valued decision samples")
    if level_zero.numel() < 2 or level_one.numel() < 2:
        raise ValueError("each decision level requires at least two samples")
    separation = (level_one.mean() - level_zero.mean()).abs()
    total_sigma = level_zero.std(correction=0) + level_one.std(correction=0)
    return separation / total_sigma.clamp_min(eps)


def gaussian_ber_estimate(level_zero: Tensor, level_one: Tensor, *, eps: float = 1e-20) -> Tensor:
    """在等先验高斯 Q 因子假设下估计二元 BER。"""
    q = gaussian_q_factor(level_zero, level_one, eps=eps)
    return 0.5 * torch.erfc(q / math.sqrt(2.0))
