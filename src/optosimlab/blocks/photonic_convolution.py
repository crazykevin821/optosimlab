"""依据 Xu 等 Nature 2021 路线、由物理器件组成的实值 WDM 卷积。"""

from __future__ import annotations

from collections.abc import Iterator
import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import SimulationGrid
from ..devices.comb import OpticalFrequencyComb
from ..devices.fiber import LinearDispersiveFiber
from ..devices.mzm import IdealMZM, MeasuredMZM, MachZehnderModulator
from ..devices.photodetector import PhotodetectorSum
from ..devices.waveshaper import WaveShaper
from .chain import OpticalChain


class PhotonicRealConvolution(nn.Module):
    """仅由物理器件组成的非负实值卷积。

    实际执行图为 OpticalFrequencyComb -> WaveShaper -> MZM ->
    LinearDispersiveFiber -> PhotodetectorSum。输入序列经所选 MZM 的标定
    反函数换算为驱动电压；光纤长度由 beta2 决定，使窄带 WDM 近似下的
    相邻梳齿恰好获得 ``tap_delay_samples`` 个样本的相对延时。
    """

    def __init__(
        self,
        power_weights: Tensor,
        *,
        sample_rate_hz: float = 100e9,
        line_spacing_hz: float = 10e9,
        line_power_w: float = 1e-3,
        tap_delay_samples: int = 1,
        dispersion_ps_nm_km: float = 17.0,
        modulator: IdealMZM | MachZehnderModulator | MeasuredMZM | None = None,
        fiber: LinearDispersiveFiber | None = None,
        detector: PhotodetectorSum | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(power_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() < 1:
            raise ValueError("power_weights must be a non-empty one-dimensional tensor")
        if torch.any((weights < 0) | (weights > 1)):
            raise ValueError("power_weights must lie in [0, 1]")
        if line_power_w <= 0:
            raise ValueError("line_power_w must be positive")
        if not isinstance(tap_delay_samples, int) or tap_delay_samples < 1:
            raise ValueError("tap_delay_samples must be a positive integer")
        if line_spacing_hz <= 0:
            raise ValueError("line_spacing_hz must be positive")

        self.line_power_w = float(line_power_w)
        self.tap_delay_samples = tap_delay_samples
        self.grid = SimulationGrid(sample_rate_hz=sample_rate_hz)
        self.comb = OpticalFrequencyComb(
            self.grid,
            line_spacing_hz=line_spacing_hz,
            line_count=weights.numel(),
            line_powers=line_power_w,
            trainable=False,
        )
        offsets = self.comb.nominal_line_offsets_hz(dtype=torch.float32)
        attenuation_db = -10.0 * torch.log10(weights.clamp_min(1e-30))
        if weights.numel() == 1:
            control_frequencies = torch.tensor([-line_spacing_hz / 2.0, line_spacing_hz / 2.0])
            attenuation_db = attenuation_db.expand(2).clone()
        else:
            control_frequencies = offsets
        waveshaper = WaveShaper(
            self.grid,
            control_frequencies_hz=control_frequencies,
            attenuation_db=attenuation_db,
            trainable=trainable,
        )
        waveshaper.phase_rad.requires_grad_(False)

        selected_modulator = modulator if modulator is not None else IdealMZM(trainable=False)
        if not hasattr(selected_modulator, "voltage_for_power_transmission") or not hasattr(
            selected_modulator, "power_transmission_bounds"
        ):
            raise TypeError("modulator must provide inverse transfer and power-transmission bounds")

        if fiber is None:
            beta2 = LinearDispersiveFiber.beta2_from_dispersion(
                dispersion_ps_nm_km, self.grid.center_wavelength_m
            )
            if beta2 >= 0:
                raise ValueError("dispersion must produce negative beta2 for increasing-frequency positive delays")
            delay_s = tap_delay_samples / sample_rate_hz
            length_m = delay_s / (-beta2 * 2.0 * math.pi * line_spacing_hz)
            selected_fiber = LinearDispersiveFiber(
                self.grid,
                length_m=length_m,
                beta2_s2_per_m=beta2,
                attenuation_db_per_km=0.0,
                trainable=False,
            )
        else:
            selected_fiber = fiber
            actual_delay = float(selected_fiber.adjacent_channel_delay_s(line_spacing_hz).detach())
            expected_delay = tap_delay_samples / sample_rate_hz
            if not math.isclose(actual_delay, expected_delay, rel_tol=1e-5, abs_tol=1e-15):
                raise ValueError("provided fiber does not match tap_delay_samples at this line spacing")

        selected_detector = detector if detector is not None else PhotodetectorSum()
        if not isinstance(selected_detector, PhotodetectorSum):
            raise TypeError("detector must inherit PhotodetectorSum")
        self.optical_chain = OpticalChain.from_graph(
            [
                ("waveshaper", waveshaper, ("comb", "offsets"), "weighted"),
                ("modulator", selected_modulator, ("weighted", "voltage"), "modulated"),
                ("fiber", selected_fiber, ("modulated", "offsets"), "delayed"),
                ("photodetector", selected_detector, "delayed", "power_w"),
            ],
            outputs="power_w",
        )

    @property
    def waveshaper(self) -> WaveShaper:
        module = self.optical_chain.graph_devices["waveshaper"]
        assert isinstance(module, WaveShaper)
        return module

    @property
    def modulator(self) -> IdealMZM | MachZehnderModulator | MeasuredMZM:
        module = self.optical_chain.graph_devices["modulator"]
        assert isinstance(module, (IdealMZM, MachZehnderModulator, MeasuredMZM))
        return module

    @property
    def fiber(self) -> LinearDispersiveFiber:
        module = self.optical_chain.graph_devices["fiber"]
        assert isinstance(module, LinearDispersiveFiber)
        return module

    @property
    def detector(self) -> PhotodetectorSum:
        module = self.optical_chain.graph_devices["photodetector"]
        assert isinstance(module, PhotodetectorSum)
        return module

    @property
    def power_weights(self) -> Tensor:
        offsets = self.comb.nominal_line_offsets_hz(
            device=self.waveshaper.attenuation_db.device,
            dtype=self.waveshaper.attenuation_db.dtype,
        )
        return self.waveshaper.power_transmission_at(offsets)

    def detected_power_w(self, samples: Tensor) -> Tensor:
        """执行物理器件图并返回单位为 W 的探测功率。"""
        if not isinstance(samples, Tensor) or torch.is_complex(samples) or not samples.is_floating_point():
            raise TypeError("samples must be a real floating-point torch.Tensor")
        if torch.any((samples.detach() < 0) | (samples.detach() > 1)):
            raise ValueError("MZM-normalized input samples must be nonnegative and at most 1")
        maximum_delay = (self.comb.line_count - 1) * self.tap_delay_samples
        padded_samples = F.pad(samples, (0, maximum_delay))
        total_samples = padded_samples.shape[-1]
        offsets = self.comb.nominal_line_offsets_hz(device=samples.device, dtype=samples.dtype)
        channels = self.comb.channel_envelopes(
            total_samples, device=samples.device, real_dtype=samples.dtype
        )
        if samples.ndim > 1:
            channels = channels.view(*([1] * (samples.ndim - 1)), *channels.shape)
        minimum, maximum = self.modulator.power_transmission_bounds(padded_samples)
        swing = maximum - minimum
        if torch.any(swing.detach() <= torch.finfo(samples.dtype).eps):
            raise ValueError("modulator must have nonzero reachable power-transmission swing")
        target_transmission = minimum + swing * padded_samples
        voltage = self.modulator.voltage_for_power_transmission(target_transmission)
        while voltage.ndim < channels.ndim:
            voltage = voltage.unsqueeze(-2)
        power = self.optical_chain({"comb": channels, "offsets": offsets, "voltage": voltage})
        return power[..., : samples.shape[-1]]

    def forward(self, samples: Tensor) -> Tensor:
        """返回经过标定、零初始状态的因果卷积。"""
        detected = self.detected_power_w(samples)
        minimum, maximum = self.modulator.power_transmission_bounds(samples)
        fiber_transmission = self.fiber.power_transmission(samples)
        weighted_line_power = self.line_power_w * self.power_weights.sum().to(
            device=samples.device, dtype=samples.dtype
        )
        baseline = fiber_transmission * minimum * weighted_line_power
        calibration = self.line_power_w * fiber_transmission * (maximum - minimum)
        if torch.any(calibration.detach() <= torch.finfo(samples.dtype).tiny):
            raise ValueError("combined line power, fibre transmission and MZM swing are too small to calibrate")
        return (detected - baseline) / calibration

    def iter_chunked(self, samples: Tensor, chunk_size: int) -> Iterator[Tensor]:
        """按序列分块执行，降低长记录的峰值内存并保持精确因果历史。

        每块自动携带 ``(tap_count-1)*tap_delay_samples`` 个历史样本，记录
        开头缺失的历史用零填充。调用方可逐块计算损失并立即反向传播，
        从而避免同时保留全部分块的 autograd 图。
        """
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if samples.shape[-1] <= chunk_size:
            yield self(samples)
            return
        maximum_delay = (self.comb.line_count - 1) * self.tap_delay_samples
        total = samples.shape[-1]
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            history_start = max(0, start - maximum_delay)
            segment = samples[..., history_start:stop]
            missing_history = maximum_delay - (start - history_start)
            if missing_history:
                segment = F.pad(segment, (missing_history, 0))
            segment_output = self(segment)
            yield segment_output[..., - (stop - start) :]

    def forward_chunked(self, samples: Tensor, chunk_size: int) -> Tensor:
        """拼接 :meth:`iter_chunked` 的输出并保持与整段前向一致。"""
        return torch.cat(tuple(self.iter_chunked(samples, chunk_size)), dim=-1)

    def project_physical_parameters_(self) -> "PhotonicRealConvolution":
        """优化器更新后把无源 WaveShaper 衰减投影回非负物理区间。"""
        with torch.no_grad():
            self.waveshaper.attenuation_db.clamp_(min=0.0)
        return self


class DifferentialPhotonicConvolution(nn.Module):
    """由两条独立正、负物理卷积支路作差得到的有符号卷积。"""

    def __init__(
        self,
        weights: Tensor,
        *,
        differential_depth: float = 0.8,
        **kwargs: object,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(weights, dtype=torch.float32)
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError("weights must be a non-empty one-dimensional tensor")
        if not 0.0 < differential_depth < 1.0:
            raise ValueError("differential_depth must lie strictly between 0 and 1")
        # 标定尺度至少取 1，避免全零或极小初始核把可训练动态范围和梯度一并压扁。
        kernel_scale = values.abs().max().clamp_min(1.0)
        normalized = values / kernel_scale
        depth = torch.tensor(float(differential_depth), dtype=values.dtype)
        self.register_buffer("differential_depth", depth)
        self.register_buffer("weight_scale", kernel_scale / depth)
        positive_kwargs = dict(kwargs)
        negative_kwargs = copy.deepcopy(kwargs)
        positive_weights = 0.5 * (1.0 + depth * normalized)
        negative_weights = 0.5 * (1.0 - depth * normalized)
        self.positive = PhotonicRealConvolution(positive_weights, **positive_kwargs)
        self.negative = PhotonicRealConvolution(negative_weights, **negative_kwargs)

    def forward(self, samples: Tensor) -> Tensor:
        return self.weight_scale * (self.positive(samples) - self.negative(samples))

    def forward_chunked(self, samples: Tensor, chunk_size: int) -> Tensor:
        """对正、负两条物理支路使用相同分块边界。"""
        return torch.cat(tuple(self.iter_chunked(samples, chunk_size)), dim=-1)

    def iter_chunked(self, samples: Tensor, chunk_size: int) -> Iterator[Tensor]:
        """逐块返回差分输出，允许调用方逐块反向传播以限制峰值内存。"""
        positive_chunks = self.positive.iter_chunked(samples, chunk_size)
        negative_chunks = self.negative.iter_chunked(samples, chunk_size)
        for positive, negative in zip(positive_chunks, negative_chunks, strict=True):
            yield self.weight_scale * (positive - negative)

    @property
    def effective_weights(self) -> Tensor:
        """返回两条 WaveShaper 支路当前表示的有符号卷积核。"""
        return self.weight_scale * (self.positive.power_weights - self.negative.power_weights)

    def project_physical_parameters_(self) -> "DifferentialPhotonicConvolution":
        self.positive.project_physical_parameters_()
        self.negative.project_physical_parameters_()
        return self
