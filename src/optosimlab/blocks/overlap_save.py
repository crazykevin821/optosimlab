"""Differentiable overlap-save filtering for long complex optical records."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..devices.base import require_complex_field


class OverlapSaveFIR(nn.Module):
    """Causal complex FIR using FFT overlap-save instead of circular convolution.

    For a causal impulse response ``h[0:M]`` and block length ``N_fft``, each
    block discards its first ``M-1`` IFFT samples and retains
    ``N_fft-M+1`` samples.  The input is zero-padded on the left by ``M-1``;
    therefore the returned record is exactly the first ``len(field)`` samples
    of the linear convolution ``field * h`` with zero initial conditions.
    ``impulse_response`` is an optionally trainable complex ``nn.Parameter``.
    """

    def __init__(self, impulse_response: Tensor, fft_size: int, *, trainable: bool = False) -> None:
        super().__init__()
        if not isinstance(impulse_response, Tensor) or not torch.is_complex(impulse_response):
            raise TypeError("impulse_response must be a one-dimensional complex torch.Tensor")
        if impulse_response.ndim != 1 or impulse_response.numel() == 0:
            raise ValueError("impulse_response must be one-dimensional and non-empty")
        if fft_size <= impulse_response.numel() - 1:
            raise ValueError("fft_size must be greater than len(impulse_response)-1")
        self.impulse_response = nn.Parameter(impulse_response.clone(), requires_grad=trainable)
        self.fft_size = int(fft_size)

    @property
    def impulse_length(self) -> int:
        """Number of causal impulse-response taps."""
        return int(self.impulse_response.numel())

    @property
    def valid_samples_per_block(self) -> int:
        """Non-corrupted output samples retained from each FFT block."""
        return self.fft_size - self.impulse_length + 1

    def frequency_response(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Return the unshifted FFT response evaluated at ``fft_size`` bins."""
        device = device or self.impulse_response.device
        if dtype is None:
            impulse = self.impulse_response.to(device=device)
        else:
            complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            impulse = self.impulse_response.to(device=device, dtype=complex_dtype)
        return torch.fft.fft(impulse, n=self.fft_size)

    def forward(self, field: Tensor) -> Tensor:
        """Filter a complex field on its final sample dimension using overlap-save."""
        require_complex_field(field)
        impulse = self.impulse_response.to(device=field.device, dtype=field.dtype)
        overlap = self.impulse_length - 1
        valid = self.valid_samples_per_block
        samples = field.shape[-1]
        blocks = (samples + valid - 1) // valid
        padded = torch.nn.functional.pad(field, (overlap, blocks * valid - samples))
        segments = padded.unfold(-1, self.fft_size, valid)
        response = torch.fft.fft(impulse, n=self.fft_size)
        transformed = torch.fft.ifft(torch.fft.fft(segments, dim=-1) * response, dim=-1)
        output = transformed[..., overlap:]
        return output.reshape(*field.shape[:-1], -1)[..., :samples]
