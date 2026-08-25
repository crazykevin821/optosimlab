"""Trainable complex-valued convolution built from real PyTorch convolutions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ComplexConv1d(nn.Module):
    """Complex 1-D convolution using four real convolutions.

    Input and output use shape ``(batch, channels, samples)`` and complex dtype.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs: object) -> None:
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, field: Tensor) -> Tensor:
        if not torch.is_complex(field):
            raise TypeError("ComplexConv1d expects a complex-valued tensor")
        # Convolution biases are added once per complex output component.
        # Calling each Conv1d directly would add them twice and mix their signs.
        real = F.conv1d(field.real, self.real.weight, None, self.real.stride, self.real.padding, self.real.dilation, self.real.groups)
        real = real - F.conv1d(field.imag, self.imag.weight, None, self.imag.stride, self.imag.padding, self.imag.dilation, self.imag.groups)
        imag = F.conv1d(field.imag, self.real.weight, None, self.real.stride, self.real.padding, self.real.dilation, self.real.groups)
        imag = imag + F.conv1d(field.real, self.imag.weight, None, self.imag.stride, self.imag.padding, self.imag.dilation, self.imag.groups)
        if self.real.bias is not None:
            real = real + self.real.bias.view(1, -1, 1)
        if self.imag.bias is not None:
            imag = imag + self.imag.bias.view(1, -1, 1)
        return torch.complex(real, imag)
