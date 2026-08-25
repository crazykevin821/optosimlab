"""Metrics for validating signal-level optical simulations."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import SimulationGrid
from .devices.base import require_complex_field


def mean_power(field: Tensor) -> Tensor:
    """Mean optical power in the simulation's field-squared units."""
    require_complex_field(field)
    return field.abs().square().mean()


def normalized_mean_square_error(reference: Tensor, estimate: Tensor, *, eps: float = 1e-12) -> Tensor:
    """NMSE = mean(|estimate-reference|^2) / mean(|reference|^2)."""
    require_complex_field(reference)
    require_complex_field(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    numerator = (estimate - reference).abs().square().mean()
    denominator = reference.abs().square().mean().clamp_min(eps)
    return numerator / denominator


def snr_db(signal: Tensor, noise: Tensor, *, eps: float = 1e-20) -> Tensor:
    """Power SNR in dB for separate complex signal and noise tensors."""
    require_complex_field(signal)
    require_complex_field(noise)
    if signal.shape != noise.shape:
        raise ValueError("signal and noise must have the same shape")
    ratio = signal.abs().square().mean() / noise.abs().square().mean().clamp_min(eps)
    return 10.0 * torch.log10(ratio)


def error_vector_magnitude(reference: Tensor, estimate: Tensor, *, eps: float = 1e-20) -> Tensor:
    """Return RMS EVM as a unitless ratio for two complex envelopes."""
    require_complex_field(reference)
    require_complex_field(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    return torch.sqrt((estimate - reference).abs().square().mean() / reference.abs().square().mean().clamp_min(eps))


def power_spectrum(field: Tensor) -> Tensor:
    """Return fftshifted per-bin power contributions that sum to mean power.

    With PyTorch's unnormalised FFT, each output bin is ``|X[k]|^2 / N^2``.
    Summing across the final dimension therefore equals ``mean(|field|^2)`` by
    Parseval's theorem.  Leading batch dimensions are preserved.
    """
    require_complex_field(field)
    samples = field.shape[-1]
    spectrum = torch.fft.fftshift(torch.fft.fft(field, dim=-1), dim=-1)
    return spectrum.abs().square() / samples**2


def centered_frequency_axis_hz(grid: SimulationGrid, samples: int, *, device: torch.device | None = None) -> Tensor:
    """Return the frequency axis aligned with :func:`power_spectrum`."""
    return torch.fft.fftshift(grid.frequencies_hz(samples, device=device))


def eye_diagram(signal: Tensor, samples_per_symbol: int, *, offset_samples: int = 0) -> Tensor:
    """Fold a signal into overlapping two-symbol eye-diagram traces.

    The returned shape is ``signal.shape[:-1] + (traces, 2*samples_per_symbol)``.
    Consecutive traces begin one symbol apart, so each UI participates in two
    adjacent eye traces (except at the record boundaries).
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
    """Return binary Gaussian decision Q from two real-valued sample classes."""
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
    """Estimate equal-prior binary BER from a Gaussian Q-factor assumption."""
    q = gaussian_q_factor(level_zero, level_one, eps=eps)
    return 0.5 * torch.erfc(q / math.sqrt(2.0))
