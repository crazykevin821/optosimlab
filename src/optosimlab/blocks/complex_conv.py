"""由 PyTorch 实卷积组成的可训练数学复值卷积。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ComplexConv1d(nn.Module):
    """使用实卷积实现的旧版数学复值一维卷积。

    输入和输出均为复数张量，形状是 ``(批次, 通道, 样本)``。
    本类仅用于兼容通用神经网络，并不代表 Xu 等人在 2021 年论文中的
    光子卷积硬件；物理 WDM 路径应使用 ``PhotonicRealConvolution``。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs: object) -> None:
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, field: Tensor) -> Tensor:
        if not torch.is_complex(field):
            raise TypeError("ComplexConv1d expects a complex-valued tensor")
        # 每个复输出分量的偏置只能加入一次；直接调用 Conv1d 会重复加入并混淆符号。
        real = F.conv1d(field.real, self.real.weight, None, self.real.stride, self.real.padding, self.real.dilation, self.real.groups)
        real = real - F.conv1d(field.imag, self.imag.weight, None, self.imag.stride, self.imag.padding, self.imag.dilation, self.imag.groups)
        imag = F.conv1d(field.imag, self.real.weight, None, self.real.stride, self.real.padding, self.real.dilation, self.real.groups)
        imag = imag + F.conv1d(field.real, self.imag.weight, None, self.imag.stride, self.imag.padding, self.imag.dilation, self.imag.groups)
        if self.real.bias is not None:
            real = real + self.real.bias.view(1, -1, 1)
        if self.imag.bias is not None:
            imag = imag + self.imag.bias.view(1, -1, 1)
        return torch.complex(real, imag)
