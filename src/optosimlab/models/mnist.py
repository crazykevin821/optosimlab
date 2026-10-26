"""用于 MNIST 小型验证的光卷积分类器。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..blocks.photonic_convolution import DifferentialPhotonicConvolution
from ..config import SimulationGrid
from ..devices.fiber import LinearDispersiveFiber
from ..devices.mzm import IntensityMZMActivation, PracticalMZM
from ..devices.photodetector import NoisyPhotodetectorSum


class PhotonicMNISTClassifier(nn.Module):
    """一层光学卷积、一个非线性层和一个电子线性分类层。

    图像按行展平为一维串行数据，批量维始终保留。``mode="ideal"`` 使用
    理想 MZM、无损光纤和无噪探测；``mode="nonideal"`` 使用 3 dB 插损、
    20 dB 消光比的实际 MZM、0.2 dB/km 光纤以及可配置的探测读出噪声。
    ``activation="relu"`` 对应 Issue 要求的电子激活，``"mzi"`` 用强度
    MZI 非线性区间替代电子激活做对照。
    """

    def __init__(
        self,
        *,
        image_size: int = 28,
        kernel_size: int = 5,
        mode: str = "ideal",
        activation: str = "relu",
        sample_rate_hz: float = 100e9,
        line_spacing_hz: float = 10e9,
        detector_noise_std_w: float = 1e-6,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if image_size < 1 or kernel_size < 1:
            raise ValueError("image_size and kernel_size must be positive")
        if mode not in {"ideal", "nonideal"}:
            raise ValueError("mode must be 'ideal' or 'nonideal'")
        if activation not in {"relu", "mzi"}:
            raise ValueError("activation must be 'relu' or 'mzi'")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive or None")

        self.image_size = int(image_size)
        self.mode = mode
        self.activation_name = activation
        self.chunk_size = chunk_size
        initial_weights = torch.linspace(-0.3, 0.3, kernel_size)
        convolution_kwargs: dict[str, object] = {
            "sample_rate_hz": sample_rate_hz,
            "line_spacing_hz": line_spacing_hz,
            "trainable": True,
        }

        if mode == "nonideal":
            grid = SimulationGrid(sample_rate_hz)
            beta2 = LinearDispersiveFiber.beta2_from_dispersion(17.0, grid.center_wavelength_m)
            length_m = (1.0 / sample_rate_hz) / (-beta2 * 2.0 * math.pi * line_spacing_hz)
            convolution_kwargs.update(
                modulator=PracticalMZM(
                    v_pi=1.0,
                    bias_voltage=0.0,
                    insertion_loss_db=3.0,
                    extinction_ratio_db=20.0,
                    trainable=False,
                ),
                fiber=LinearDispersiveFiber(
                    grid,
                    length_m=length_m,
                    attenuation_db_per_km=0.2,
                    beta2_s2_per_m=beta2,
                    trainable=False,
                ),
                detector=NoisyPhotodetectorSum(noise_std_w=detector_noise_std_w),
            )

        self.optical_convolution = DifferentialPhotonicConvolution(initial_weights, **convolution_kwargs)
        self.activation = nn.ReLU() if activation == "relu" else IntensityMZMActivation()
        self.classifier = nn.Linear(self.image_size * self.image_size, 10)

    @property
    def nonideality_description(self) -> dict[str, float | str]:
        """返回实验配置中实际启用的非理想参数，便于记录复现实验。"""
        if self.mode == "ideal":
            return {"mode": "ideal", "fiber_loss_db_per_km": 0.0, "detector_noise_std_w": 0.0}
        detector = self.optical_convolution.positive.detector
        return {
            "mode": "nonideal",
            "mzm_insertion_loss_db": 3.0,
            "mzm_extinction_ratio_db": 20.0,
            "fiber_loss_db_per_km": 0.2,
            "detector_noise_std_w": float(detector.noise_std_w.detach()),  # type: ignore[attr-defined]
        }

    def forward(self, images: Tensor) -> Tensor:
        if not isinstance(images, Tensor) or torch.is_complex(images) or not images.is_floating_point():
            raise TypeError("images must be a real floating-point torch.Tensor")
        if images.ndim == 4:
            if images.shape[1:] != (1, self.image_size, self.image_size):
                raise ValueError("four-dimensional input must have shape (batch, 1, image_size, image_size)")
            serial = images.flatten(start_dim=1)
        elif images.ndim == 2 and images.shape[1] == self.image_size * self.image_size:
            serial = images
        else:
            raise ValueError("images must be a flattened batch or a batch of single-channel square images")
        if torch.any((serial.detach() < 0) | (serial.detach() > 1)):
            raise ValueError("MNIST input pixels must be normalized to [0, 1]")

        if self.chunk_size is None:
            features = self.optical_convolution(serial)
        else:
            features = self.optical_convolution.forward_chunked(serial, self.chunk_size)
        return self.classifier(self.activation(features))

    def project_physical_parameters_(self) -> "PhotonicMNISTClassifier":
        """在 ``optimizer.step()`` 后投影无源器件参数。"""
        self.optical_convolution.project_physical_parameters_()
        if isinstance(self.activation, IntensityMZMActivation):
            self.activation.project_physical_parameters_()
        return self
