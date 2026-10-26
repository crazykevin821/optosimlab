"""用于长复光场记录的可微 overlap-save 滤波。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..devices.base import require_complex_field


class OverlapSaveFIR(nn.Module):
    """使用 FFT overlap-save、避免循环卷积的因果复 FIR。

    对因果冲激响应 ``h[0:M]`` 和块长 ``N_fft``，每块舍弃前 ``M-1`` 个
    IFFT 样本并保留 ``N_fft-M+1`` 个样本。输入左侧补 ``M-1`` 个零，
    因而返回结果严格等于零初始条件下线性卷积 ``field * h`` 的前
    ``len(field)`` 个样本。``impulse_response`` 可选为可训练复参数。
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
        """因果冲激响应抽头数。"""
        return int(self.impulse_response.numel())

    @property
    def valid_samples_per_block(self) -> int:
        """每个 FFT 块保留的无循环污染输出样本数。"""
        return self.fft_size - self.impulse_length + 1

    def frequency_response(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """返回在 ``fft_size`` 个频点计算的未移位 FFT 响应。"""
        device = device or self.impulse_response.device
        if dtype is None:
            impulse = self.impulse_response.to(device=device)
        else:
            complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            impulse = self.impulse_response.to(device=device, dtype=complex_dtype)
        return torch.fft.fft(impulse, n=self.fft_size)

    def forward(self, field: Tensor) -> Tensor:
        """沿复光场最后一维使用 overlap-save 进行滤波。"""
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
