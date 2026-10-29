"""基于 MZI 复权重与相干求和的物理复值一维卷积。"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..devices.mzm import MZIComplexWeightBank


class MZIComplexConv1d(nn.Module):
    """MZI 参数化的因果复值光学卷积。

    输入形状为 ``(batch, in_channels, samples)``，也接受省略 batch 的
    ``(in_channels, samples)``。输出满足

    ``y_o[n] = sum_c sum_k w[o,c,k] * x_c[n-kD] + b_o``，

    其中 ``D=tap_delay_samples``。复系数由 :class:`MZIComplexWeightBank`
    的内部相移 ``theta`` 和外部相移 ``phi`` 产生，随后对各光路做相干
    场求和；这不同于平方律探测后的实值 WDM 功率卷积。
    """

    def __init__(
        self,
        complex_weights: Tensor,
        *,
        tap_delay_samples: int = 1,
        bias: bool = False,
        insertion_loss_db: float = 0.0,
        phase_error_std_rad: float = 0.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(complex_weights)
        if not torch.is_complex(weights) or weights.ndim != 3:
            raise ValueError("complex_weights must have shape (out_channels, in_channels, kernel_size)")
        if not isinstance(tap_delay_samples, int) or tap_delay_samples < 1:
            raise ValueError("tap_delay_samples must be a positive integer")
        self.tap_delay_samples = tap_delay_samples
        self.weight_bank = MZIComplexWeightBank(
            weights,
            insertion_loss_db=insertion_loss_db,
            phase_error_std_rad=phase_error_std_rad,
            trainable=trainable,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(weights.shape[0], dtype=weights.dtype), requires_grad=trainable)
        else:
            self.register_parameter("bias", None)

    @property
    def in_channels(self) -> int:
        return self.weight_bank.in_channels

    @property
    def out_channels(self) -> int:
        return self.weight_bank.out_channels

    @property
    def kernel_size(self) -> int:
        return self.weight_bank.kernel_size

    @property
    def maximum_delay(self) -> int:
        return (self.kernel_size - 1) * self.tap_delay_samples

    @property
    def effective_weights(self) -> Tensor:
        """返回当前 MZI 相移对应的复卷积核。"""
        return self.weight_bank.effective_weights()

    def _forward_with_weights(self, field: Tensor, effective_weights: Tensor) -> Tensor:
        if not isinstance(field, Tensor) or not torch.is_complex(field):
            raise TypeError("field must be a complex-valued torch.Tensor")
        unbatched = field.ndim == 2
        if unbatched:
            field = field.unsqueeze(0)
        if field.ndim != 3 or field.shape[1] != self.in_channels:
            raise ValueError("field must have shape (batch, in_channels, samples)")

        padded = F.pad(field, (self.maximum_delay, 0))
        windows = padded.unfold(-1, self.maximum_delay + 1, 1)
        windows = windows[..., :: self.tap_delay_samples].flip(-1)
        output = self.weight_bank(windows, effective_weights)
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype).view(1, -1, 1)
        return output.squeeze(0) if unbatched else output

    def forward(self, field: Tensor) -> Tensor:
        """采样一次器件相位误差，并用于整条记录。"""
        return self._forward_with_weights(field, self.weight_bank.effective_weights())

    def iter_chunked(self, field: Tensor, chunk_size: int) -> Iterator[Tensor]:
        """逐块返回因果复卷积，允许逐块反向传播以限制峰值内存。

        一次迭代复用同一份器件相移误差，但每块重新构造参数到权重的
        autograd 图，因此各块可以依次调用 ``backward()``。
        """
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(field, Tensor) or not torch.is_complex(field):
            raise TypeError("field must be a complex-valued torch.Tensor")
        if field.ndim not in (2, 3) or field.shape[-1] < 1:
            raise ValueError(
                "field must have shape (in_channels, samples) or "
                "(batch, in_channels, samples), with at least one sample"
            )
        phase_perturbation = self.weight_bank.sample_phase_perturbation()
        if field.shape[-1] <= chunk_size:
            yield self._forward_with_weights(
                field, self.weight_bank.effective_weights(phase_perturbation)
            )
            return
        total = field.shape[-1]
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            history_start = max(0, start - self.maximum_delay)
            segment = field[..., history_start:stop]
            missing_history = self.maximum_delay - (start - history_start)
            if missing_history:
                segment = F.pad(segment, (missing_history, 0))
            # 每块重建参数到权重的 autograd 图，但复用同一次逻辑前向的相移误差。
            effective_weights = self.weight_bank.effective_weights(phase_perturbation)
            segment_output = self._forward_with_weights(segment, effective_weights)
            yield segment_output[..., - (stop - start) :]

    def forward_chunked(self, field: Tensor, chunk_size: int) -> Tensor:
        """拼接分块输出，结果与一次性因果卷积一致并保留 autograd。"""
        return torch.cat(tuple(self.iter_chunked(field, chunk_size)), dim=-1)

    def project_physical_parameters_(self) -> "MZIComplexConv1d":
        """把复权重单元投影回无源可实现集合。"""
        self.weight_bank.project_physical_parameters_()
        return self
