"""确定性小信号掺铒光纤放大器模型。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import C_LIGHT, H_PLANCK, SimulationGrid
from ..units import db_to_power_ratio, dbm_to_w
from .base import GraphDevice, nonnegative_parameter, require_complex_field


class GainControlledEDFA(GraphDevice):
    """按商用设备方式直接设定增益的 EDFA。

    面向用户的控制量是峰值净功率增益 dB，对应商用放大器的恒增益模式。
    泵浦电流及粒子数动态不在此链路级模型范围内。

    未饱和功率增益为高斯形，其半高全宽为 ``gain_bandwidth_hz``。
    饱和采用以输出功率为参考的平滑转折：

        G_eff = 1 + (G_ss - 1) / (1 + G_ss * P_in / P_sat,out)

    当预测小信号输出达到 ``P_sat,out`` 时，超出 1 的增益部分约压缩 3 dB，
    同时总增益不低于 1。ASE 使用单偏振复包络约定。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        gain_db: float = 20.0,
        gain_bandwidth_hz: float = 4e12,
        gain_center_offset_hz: float = 0.0,
        output_saturation_power_dbm: float = 20.0,
        noise_figure_db: float = 5.0,
        *,
        enable_saturation: bool = True,
        enable_ase: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if gain_db < 0:
            raise ValueError("gain_db must not be negative")
        if gain_bandwidth_hz <= 0:
            raise ValueError("gain_bandwidth_hz must be positive")
        if noise_figure_db < 0:
            raise ValueError("noise_figure_db must not be negative")
        self.grid = grid
        self.gain_db = nn.Parameter(torch.tensor(float(gain_db)), requires_grad=trainable)
        self.gain_bandwidth_hz = nn.Parameter(torch.tensor(float(gain_bandwidth_hz)), requires_grad=trainable)
        self.gain_center_offset_hz = nn.Parameter(torch.tensor(float(gain_center_offset_hz)), requires_grad=trainable)
        self.output_saturation_power_dbm = nn.Parameter(
            torch.tensor(float(output_saturation_power_dbm)), requires_grad=trainable
        )
        self.noise_figure_db = nn.Parameter(torch.tensor(float(noise_figure_db)), requires_grad=trainable)
        self.enable_saturation = bool(enable_saturation)
        self.enable_ase = bool(enable_ase)

    def small_signal_power_gain(
        self,
        samples: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """返回 FFT 顺序的未饱和功率增益。"""
        if samples <= 0:
            raise ValueError("samples must be positive")
        device = device or self.gain_db.device
        dtype = dtype or self.gain_db.dtype
        gain_db = nonnegative_parameter(self.gain_db, "gain_db").to(device=device, dtype=dtype)
        bandwidth = nonnegative_parameter(
            self.gain_bandwidth_hz,
            "gain_bandwidth_hz",
            floor=torch.finfo(dtype).eps,
        ).to(device=device, dtype=dtype)
        centre = self.gain_center_offset_hz.to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        profile = torch.exp(-4.0 * math.log(2.0) * ((frequency - centre) / bandwidth).square())
        return db_to_power_ratio(gain_db).to(device=device, dtype=dtype) * profile

    def effective_power_gain(self, field: Tensor) -> Tensor:
        """返回经过可选输出参考饱和后的增益。"""
        require_complex_field(field)
        gain = self.small_signal_power_gain(
            field.shape[-1], device=field.device, dtype=field.real.dtype
        )
        if not self.enable_saturation:
            return gain
        input_power_w = field.abs().square().mean(dim=-1, keepdim=True)
        saturation_w = dbm_to_w(
            self.output_saturation_power_dbm.to(device=field.device, dtype=field.real.dtype)
        )
        excess_gain = (gain - 1.0).clamp_min(0.0)
        saturated_amplifying_band = 1.0 + excess_gain / (1.0 + gain * input_power_w / saturation_w)
        return torch.where(gain >= 1.0, saturated_amplifying_band, gain)

    def ase_power_spectral_density(self, gain: Tensor) -> Tensor:
        """返回单偏振输出 ASE 功率谱密度，单位 W/Hz。"""
        dtype = gain.dtype
        device = gain.device
        noise_figure_db = nonnegative_parameter(self.noise_figure_db, "noise_figure_db").to(
            device=device, dtype=dtype
        )
        n_sp = db_to_power_ratio(noise_figure_db).to(device=device, dtype=dtype) / 2.0
        photon_energy = torch.tensor(
            H_PLANCK * C_LIGHT / self.grid.center_wavelength_m,
            device=device,
            dtype=dtype,
        )
        return n_sp * photon_energy * (gain - 1.0).clamp_min(0.0)

    def forward(self, field: Tensor) -> Tensor:
        """对复光场施加增益、饱和与可选 ASE。"""
        require_complex_field(field)
        samples = field.shape[-1]
        gain = self.effective_power_gain(field)
        signal = torch.fft.ifft(torch.fft.fft(field, dim=-1) * torch.sqrt(gain), dim=-1)
        if not self.enable_ase:
            return signal
        psd = self.ase_power_spectral_density(gain)
        bin_variance = samples**2 * psd * (self.grid.sample_rate_hz / samples)
        quadrature_std = torch.sqrt(bin_variance / 2.0)
        spectral_noise = torch.complex(
            torch.randn_like(field.real) * quadrature_std,
            torch.randn_like(field.real) * quadrature_std,
        )
        return signal + torch.fft.ifft(spectral_noise, dim=-1)


class SmallSignalEDFA(GraphDevice):
    """随频率变化的确定性小信号 EDFA。

    本模型刻意只覆盖线性工作区，其峰值净功率增益 dB 为

    ``G_peak_dB = gain_slope_dB_per_mA * max(I_pump - I_transparency, 0) - L``.

    以 ``gain_center_offset_hz`` 为中心的相对功率增益为高斯形，
    半高全宽是 ``gain_bandwidth_hz``：

    ``G_power(f) = 10**(G_peak_dB / 10) * exp[-4 ln(2) ((f-f0)/B)^2]``.

    因此在 ``f0 +/- B/2`` 处，功率增益恰为峰值的一半（相对下降
    -3.0103 dB），光场乘以 ``sqrt(G_power)``。该版本不包含 ASE、
    增益饱和、泵浦动态、偏振相关性或瞬态粒子数动态。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        pump_current_ma: float = 100.0,
        transparency_current_ma: float = 20.0,
        gain_slope_db_per_ma: float = 0.25,
        gain_bandwidth_hz: float = 4e12,
        gain_center_offset_hz: float = 0.0,
        insertion_loss_db: float = 0.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if pump_current_ma < 0:
            raise ValueError("pump_current_ma must not be negative")
        if transparency_current_ma < 0:
            raise ValueError("transparency_current_ma must not be negative")
        if gain_slope_db_per_ma < 0:
            raise ValueError("gain_slope_db_per_ma must not be negative")
        if gain_bandwidth_hz <= 0:
            raise ValueError("gain_bandwidth_hz must be positive")
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.grid = grid
        self.pump_current_ma = nn.Parameter(torch.tensor(float(pump_current_ma)), requires_grad=trainable)
        self.transparency_current_ma = nn.Parameter(torch.tensor(float(transparency_current_ma)), requires_grad=trainable)
        self.gain_slope_db_per_ma = nn.Parameter(torch.tensor(float(gain_slope_db_per_ma)), requires_grad=trainable)
        self.gain_bandwidth_hz = nn.Parameter(torch.tensor(float(gain_bandwidth_hz)), requires_grad=trainable)
        self.gain_center_offset_hz = nn.Parameter(torch.tensor(float(gain_center_offset_hz)), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def peak_net_gain_db(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """返回由电流经验映射得到的峰值净功率增益 dB。"""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        pump = nonnegative_parameter(self.pump_current_ma, "pump_current_ma").to(device=device, dtype=dtype)
        transparency = nonnegative_parameter(self.transparency_current_ma, "transparency_current_ma").to(device=device, dtype=dtype)
        slope = nonnegative_parameter(self.gain_slope_db_per_ma, "gain_slope_db_per_ma").to(device=device, dtype=dtype)
        loss = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=device, dtype=dtype)
        return slope * (pump - transparency).clamp_min(0.0) - loss

    def power_gain(self, samples: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """返回有限记录对应的 FFT 顺序功率增益。

        ``gain_bandwidth_hz`` 是相对功率增益的半高全宽，不是场幅宽，
        也不是在任意 dB 电平处定义的带宽。
        """
        if samples <= 0:
            raise ValueError("samples must be positive")
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        bandwidth = nonnegative_parameter(self.gain_bandwidth_hz, "gain_bandwidth_hz", floor=torch.finfo(dtype).eps).to(device=device, dtype=dtype)
        centre = self.gain_center_offset_hz.to(device=device, dtype=dtype)
        frequency = self.grid.frequencies_hz(samples, device=device).to(dtype=dtype)
        relative_gain = torch.exp(-4.0 * math.log(2.0) * ((frequency - centre) / bandwidth).square())
        ten = torch.tensor(10.0, device=device, dtype=dtype)
        return torch.pow(ten, self.peak_net_gain_db(device=device, dtype=dtype) / 10.0) * relative_gain

    def transfer_function(self, samples: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """返回实数场幅度传输函数 ``sqrt(G_power)``。"""
        return torch.sqrt(self.power_gain(samples, device=device, dtype=dtype))

    def forward(self, field: Tensor) -> Tensor:
        """施加确定性的频率相关小信号放大。"""
        require_complex_field(field)
        transfer = self.transfer_function(field.shape[-1], device=field.device, dtype=field.real.dtype)
        return torch.fft.ifft(torch.fft.fft(field, dim=-1) * transfer, dim=-1)


class SaturatedNoisyEDFA(SmallSignalEDFA):
    """在小信号 EDFA 上增加现象学饱和与 ASE。

    对平均输入信号功率 ``P_in``，确定性增益为
    ``G_eff(f) = G_ss(f) / (1 + P_in / P_sat)``.  This is a scalar-gain
    这是标量增益饱和近似：在降低增益电平时保持频谱形状，适用于可微链路级
    实验，不适用于粒子数动态建模。

    ASE 采用单偏振复包络约定。令 ``NF_linear=10**(NF_dB/10)`` 且
    ``n_sp=NF_linear/2``，双边输出 PSD 为
    ``S_ASE(f)=n_sp*h*nu*max(G_eff(f)-1,0)``，单位 W/Hz。FFT 点数为 ``N``、
    频点宽度为 ``df`` 时，未归一化 FFT 噪声频点方差为
    ``N**2*S_ASE*df``；PyTorch IFFT 给出的时域期望平均噪声功率因而为
    ``sum(S_ASE*df)``。
    """

    def __init__(
        self,
        grid: SimulationGrid,
        pump_current_ma: float = 100.0,
        transparency_current_ma: float = 20.0,
        gain_slope_db_per_ma: float = 0.25,
        gain_bandwidth_hz: float = 4e12,
        gain_center_offset_hz: float = 0.0,
        insertion_loss_db: float = 0.0,
        saturation_power: float = 1.0,
        noise_figure_db: float = 5.0,
        *,
        enable_ase: bool = True,
        trainable: bool = True,
    ) -> None:
        if saturation_power <= 0:
            raise ValueError("saturation_power must be positive")
        if noise_figure_db < 0:
            raise ValueError("noise_figure_db must not be negative")
        super().__init__(
            grid,
            pump_current_ma=pump_current_ma,
            transparency_current_ma=transparency_current_ma,
            gain_slope_db_per_ma=gain_slope_db_per_ma,
            gain_bandwidth_hz=gain_bandwidth_hz,
            gain_center_offset_hz=gain_center_offset_hz,
            insertion_loss_db=insertion_loss_db,
            trainable=trainable,
        )
        self.saturation_power = nn.Parameter(torch.tensor(float(saturation_power)), requires_grad=trainable)
        self.noise_figure_db = nn.Parameter(torch.tensor(float(noise_figure_db)), requires_grad=trainable)
        self.enable_ase = bool(enable_ase)

    def effective_power_gain(
        self,
        samples: int,
        signal_power: Tensor | float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """返回饱和功率增益；前导维与 ``signal_power`` 一致。"""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        power = torch.as_tensor(signal_power, device=device, dtype=dtype)
        saturation = nonnegative_parameter(self.saturation_power, "saturation_power", floor=torch.finfo(dtype).eps).to(device=device, dtype=dtype)
        return self.power_gain(samples, device=device, dtype=dtype) / (1.0 + power / saturation)

    def ase_power_spectral_density(
        self,
        samples: int,
        signal_power: Tensor | float = 0.0,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """返回单偏振复包络 ASE 功率谱密度，单位 W/Hz。"""
        device = device or self.pump_current_ma.device
        dtype = dtype or self.pump_current_ma.dtype
        gain = self.effective_power_gain(samples, signal_power, device=device, dtype=dtype)
        noise_figure = nonnegative_parameter(self.noise_figure_db, "noise_figure_db").to(device=device, dtype=dtype)
        n_sp = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), noise_figure / 10.0) / 2.0
        photon_energy = torch.tensor(H_PLANCK * C_LIGHT / self.grid.center_wavelength_m, device=device, dtype=dtype)
        return n_sp * photon_energy * (gain - 1.0).clamp_min(0.0)

    def forward(self, field: Tensor) -> Tensor:
        """对 ``field`` 进行饱和放大，并可选加入统计正确的 ASE。"""
        require_complex_field(field)
        samples = field.shape[-1]
        signal_power = field.abs().square().mean(dim=-1, keepdim=True)
        gain = self.effective_power_gain(samples, signal_power, device=field.device, dtype=field.real.dtype)
        signal = torch.fft.ifft(torch.fft.fft(field, dim=-1) * torch.sqrt(gain), dim=-1)
        if not self.enable_ase:
            return signal

        psd = self.ase_power_spectral_density(samples, signal_power, device=field.device, dtype=field.real.dtype)
        bin_width_hz = self.grid.sample_rate_hz / samples
        bin_variance = samples**2 * psd * bin_width_hz
        quadrature_std = torch.sqrt(bin_variance / 2.0)
        spectral_noise = torch.complex(
            torch.randn_like(field.real) * quadrature_std,
            torch.randn_like(field.real) * quadrature_std,
        )
        return signal + torch.fft.ifft(spectral_noise, dim=-1)
