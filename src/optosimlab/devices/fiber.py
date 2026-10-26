"""线性色散单模光纤模型。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import C_LIGHT, SimulationGrid
from .base import GraphDevice, nonnegative_parameter, require_complex_field


class LinearDispersiveFiber(GraphDevice):
    """含功率衰减和二阶色散的频域光纤。

    本包的正向 FFT 采用 ``exp(-j2*pi*f*t)``，逆 FFT 采用
    ``exp(+j2*pi*f*t)``。去除 beta0 与 beta1 后，传播为
    ``H(f)=exp(-alpha_power*L/2) exp(+j*beta2*(2*pi*f)^2*L/2)``。
    ``alpha_power`` 单位为 Np/m，``beta2`` 单位为 s^2/m；传入的
    dB/km 是功率损耗而不是场幅损耗。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        length_m: float,
        *,
        attenuation_db_per_km: float = 0.0,
        beta2_s2_per_m: float | None = None,
        dispersion_ps_nm_km: float | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if length_m < 0:
            raise ValueError("length_m must not be negative")
        if attenuation_db_per_km < 0:
            raise ValueError("attenuation_db_per_km must not be negative")
        if beta2_s2_per_m is not None and dispersion_ps_nm_km is not None:
            raise ValueError("provide beta2_s2_per_m or dispersion_ps_nm_km, not both")
        if dispersion_ps_nm_km is not None:
            beta2_s2_per_m = self.beta2_from_dispersion(dispersion_ps_nm_km, grid.center_wavelength_m)
        self.grid = grid
        self.length_m = nn.Parameter(torch.tensor(float(length_m)), requires_grad=trainable)
        self.attenuation_db_per_km = nn.Parameter(torch.tensor(float(attenuation_db_per_km)), requires_grad=trainable)
        self.beta2_s2_per_m = nn.Parameter(torch.tensor(float(beta2_s2_per_m or 0.0)), requires_grad=trainable)

    @staticmethod
    def beta2_from_dispersion(dispersion_ps_nm_km: float, wavelength_m: float) -> float:
        """把 D [ps/(nm km)] 换算为 beta2 [s^2/m]。"""
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        d_s_per_m2 = dispersion_ps_nm_km * 1e-6
        return -d_s_per_m2 * wavelength_m**2 / (2.0 * math.pi * C_LIGHT)

    @staticmethod
    def dispersion_from_beta2(beta2_s2_per_m: float, wavelength_m: float) -> float:
        """把 beta2 [s^2/m] 换算为 D [ps/(nm km)]。"""
        if wavelength_m <= 0:
            raise ValueError("wavelength_m must be positive")
        return -2.0 * math.pi * C_LIGHT * beta2_s2_per_m / wavelength_m**2 / 1e-6

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        length = nonnegative_parameter(self.length_m, "length_m").to(device=device, dtype=dtype)
        attenuation_db = nonnegative_parameter(self.attenuation_db_per_km, "attenuation_db_per_km").to(device=device, dtype=dtype)
        beta2 = self.beta2_s2_per_m.to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        omega = 2.0 * torch.pi * frequency
        # 功率由 dB/km 换算为 Np/m；场幅度使用一半指数。
        alpha_power_np_per_m = attenuation_db * (math.log(10.0) / 10.0) / 1000.0
        log_amplitude = -0.5 * alpha_power_np_per_m * length
        phase = 0.5 * beta2 * omega.square() * length
        return torch.exp(torch.complex(log_amplitude.expand_as(phase), phase))

    def wdm_transfer_function(
        self,
        samples: int,
        carrier_offsets_hz: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        reference_offset_hz: Tensor | float | None = None,
    ) -> Tensor:
        """返回含波长相关延时的窄带 WDM 传输。

        在载波频偏 ``f_k`` 处展开光纤相位，可得基带线性项
        ``beta2*L*(2*pi*f_k)*(2*pi*f)``。以第一个或指定载波为参考移除公共
        群时延；在本包 FFT 符号约定下，相邻通道时延为
        ``-beta2*L*2*pi*Delta_f``。这里有意省略各通道共有的二次项，
        用来表示横向光子处理器所需的渐进延时工作区。
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        if not isinstance(carrier_offsets_hz, Tensor) or carrier_offsets_hz.ndim != 1:
            raise ValueError("carrier_offsets_hz must be a one-dimensional tensor")
        offsets = carrier_offsets_hz.to(device=device, dtype=dtype)
        reference = offsets[0] if reference_offset_hz is None else torch.as_tensor(reference_offset_hz, device=device, dtype=dtype)
        relative_carrier_omega = 2.0 * torch.pi * (offsets - reference)
        baseband_omega = 2.0 * torch.pi * self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        length = nonnegative_parameter(self.length_m, "length_m").to(device=device, dtype=dtype)
        attenuation_db = nonnegative_parameter(self.attenuation_db_per_km, "attenuation_db_per_km").to(
            device=device, dtype=dtype
        )
        beta2 = self.beta2_s2_per_m.to(device=device, dtype=dtype)
        alpha_power_np_per_m = attenuation_db * (math.log(10.0) / 10.0) / 1000.0
        log_amplitude = -0.5 * alpha_power_np_per_m * length
        phase = beta2 * length * relative_carrier_omega.unsqueeze(-1) * baseband_omega.unsqueeze(0)
        return torch.exp(torch.complex(log_amplitude.expand_as(phase), phase))

    def adjacent_channel_delay_s(self, line_spacing_hz: float) -> Tensor:
        """返回频率递增相邻通道之间的群时延增量。"""
        if line_spacing_hz <= 0:
            raise ValueError("line_spacing_hz must be positive")
        return -self.beta2_s2_per_m * self.length_m * (2.0 * math.pi * line_spacing_hz)

    def power_transmission(self, reference: Tensor) -> Tensor:
        """返回光纤与波长无关的公共功率透射率。

        ``attenuation_db_per_km`` 是功率损耗，``length_m`` 会换算为 km，故
        结果为 ``10**(-alpha_db_per_km*L_km/10)``。返回标量保留与可训练
        光纤参数之间的 autograd 连接。
        """
        if not isinstance(reference, Tensor) or torch.is_complex(reference) or not reference.is_floating_point():
            raise TypeError("reference must be a real floating-point torch.Tensor")
        length = nonnegative_parameter(self.length_m, "length_m").to(
            device=reference.device, dtype=reference.dtype
        )
        attenuation = nonnegative_parameter(
            self.attenuation_db_per_km, "attenuation_db_per_km"
        ).to(device=reference.device, dtype=reference.dtype)
        ten = torch.tensor(10.0, device=reference.device, dtype=reference.dtype)
        return torch.pow(ten, -attenuation * (length / 1000.0) / 10.0)

    def forward(self, field: Tensor, carrier_offsets_hz: Tensor | None = None) -> Tensor:
        require_complex_field(field)
        if carrier_offsets_hz is not None:
            if field.ndim < 2 or field.shape[-2] != carrier_offsets_hz.numel():
                raise ValueError("WDM field must have shape (..., channels, samples) matching carrier_offsets_hz")
            response = self.wdm_transfer_function(
                field.shape[-1], carrier_offsets_hz, device=field.device, dtype=field.real.dtype
            )
            while response.ndim < field.ndim:
                response = response.unsqueeze(0)
            return torch.fft.ifft(torch.fft.fft(field, dim=-1) * response, dim=-1)
        response = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.fft(field, dim=-1) * response, dim=-1)
