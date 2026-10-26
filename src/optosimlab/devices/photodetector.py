"""用于 WDM 光场的平方律光电探测器模型。"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import GraphDevice, nonnegative_parameter, require_complex_field


class PhotodetectorSum(GraphDevice):
    """分别探测 WDM 通道并以 W 为单位求和光功率。

    模型假设电带宽低于光梳间隔，因此滤除光学通道间拍频项。默认返回
    单位为 W 的功率；调用 ``current_a()`` 时才乘响应度并返回单位为 A 的电流。
    """

    def __init__(self, responsivity_a_per_w: float = 1.0, *, trainable: bool = False) -> None:
        super().__init__()
        if responsivity_a_per_w < 0:
            raise ValueError("responsivity_a_per_w must not be negative")
        self.responsivity_a_per_w = nn.Parameter(
            torch.tensor(float(responsivity_a_per_w)), requires_grad=trainable
        )

    def forward(self, channel_fields: Tensor) -> Tensor:
        require_complex_field(channel_fields)
        if channel_fields.ndim < 2:
            raise ValueError("channel_fields must have shape (..., channels, samples)")
        return channel_fields.abs().square().sum(dim=-2)

    def current_a(self, channel_fields: Tensor) -> Tensor:
        power_w = self(channel_fields)
        responsivity = nonnegative_parameter(self.responsivity_a_per_w, "responsivity_a_per_w").to(
            power_w.device, power_w.dtype
        )
        return responsivity * power_w


class NoisyPhotodetectorSum(PhotodetectorSum):
    """带加性高斯读出噪声的 WDM 功率探测器。

    ``noise_std_w`` 是探测器功率读数的每采样点标准差，单位 W。噪声加在
    平方律探测和通道求和之后，因此表示探测器/TIA/ADC 的等效读出噪声，
    不表示光场上的 ASE。设为零时严格退化为 :class:`PhotodetectorSum`。
    """

    def __init__(
        self,
        responsivity_a_per_w: float = 1.0,
        noise_std_w: float = 0.0,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__(responsivity_a_per_w, trainable=trainable)
        if noise_std_w < 0:
            raise ValueError("noise_std_w must not be negative")
        self.noise_std_w = nn.Parameter(torch.tensor(float(noise_std_w)), requires_grad=trainable)

    def forward(self, channel_fields: Tensor) -> Tensor:
        power = super().forward(channel_fields)
        noise_std = nonnegative_parameter(self.noise_std_w, "noise_std_w").to(
            device=power.device, dtype=power.dtype
        )
        if float(noise_std.detach()) == 0.0:
            return power
        return power + torch.randn_like(power) * noise_std


class BalancedPhotodetector(GraphDevice):
    """对正、负 WDM 支路的平方律探测功率作差。"""

    graph_input_ports = ("positive", "negative")

    def __init__(self, responsivity_a_per_w: float = 1.0, *, trainable: bool = False) -> None:
        super().__init__()
        self.positive_detector = PhotodetectorSum(responsivity_a_per_w, trainable=trainable)
        self.negative_detector = PhotodetectorSum(responsivity_a_per_w, trainable=trainable)

    def forward(self, positive_fields: Tensor, negative_fields: Tensor) -> Tensor:
        return self.positive_detector(positive_fields) - self.negative_detector(negative_fields)
