"""Integrated MZM, splitter, coupler and multi-port interferometer models."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import GraphDevice, nonnegative_parameter, require_complex_field
from .filters import FrequencyDomainLowPass


def _ratio_logit(splitting_ratio: float) -> Tensor:
    if not 0.0 < splitting_ratio < 1.0:
        raise ValueError("splitting_ratio must be strictly between 0 and 1")
    return torch.logit(torch.tensor(float(splitting_ratio)))


def _amplitude_loss(loss_db: Tensor, reference: Tensor) -> Tensor:
    checked = nonnegative_parameter(loss_db, "insertion_loss_db").to(
        device=reference.device, dtype=reference.real.dtype
    )
    ten = torch.tensor(10.0, device=reference.device, dtype=reference.real.dtype)
    return torch.pow(ten, -checked / 20.0)


def _require_real_voltage(voltage: Tensor) -> None:
    if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
        raise TypeError("voltage must be a real-valued torch.Tensor")


class PowerSplitter1x2(GraphDevice):
    """Reciprocal 1x2 splitter with explicit cross-arm quadrature phase."""

    graph_output_ports = ("through", "cross")

    def __init__(self, splitting_ratio: float = 0.5, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.splitting_logit = nn.Parameter(_ratio_logit(splitting_ratio), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    @property
    def splitting_ratio(self) -> Tensor:
        return torch.sigmoid(self.splitting_logit)

    def forward(self, field: Tensor) -> tuple[Tensor, Tensor]:
        require_complex_field(field)
        ratio = self.splitting_ratio.to(device=field.device, dtype=field.real.dtype)
        loss = _amplitude_loss(self.insertion_loss_db, field)
        return loss * torch.sqrt(ratio) * field, 1j * loss * torch.sqrt(1.0 - ratio) * field


class DirectionalCoupler2x2(GraphDevice):
    """Reciprocal unitary 2x2 coupler with j on both cross coefficients."""

    graph_input_ports = ("input0", "input1")
    graph_output_ports = ("output0", "output1")

    def __init__(self, splitting_ratio: float = 0.5, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.splitting_logit = nn.Parameter(_ratio_logit(splitting_ratio), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    @property
    def splitting_ratio(self) -> Tensor:
        return torch.sigmoid(self.splitting_logit)

    def forward(self, input0: Tensor, input1: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if input1 is None:
            require_complex_field(input0)
            if input0.ndim < 2 or input0.shape[-2] != 2:
                raise ValueError("a stacked coupler input must have shape (..., 2, samples)")
            input0, input1 = input0.unbind(dim=-2)
        require_complex_field(input0)
        require_complex_field(input1)
        if input0.shape != input1.shape:
            raise ValueError("the two coupler input ports must have identical shapes")
        ratio = self.splitting_ratio.to(device=input0.device, dtype=input0.real.dtype)
        through = torch.sqrt(ratio)
        cross = torch.sqrt(1.0 - ratio)
        loss = _amplitude_loss(self.insertion_loss_db, input0)
        return (
            loss * (through * input0 + 1j * cross * input1),
            loss * (1j * cross * input0 + through * input1),
        )


class DifferentialDualPortMZM(GraphDevice):
    """One-input, complementary-two-output push-pull MZM.

    The device is assembled from a 1x2 splitter, differential arm phase and a
    2x2 output coupler.  At 50:50 and zero loss its outputs are
    E_bar=E_in*cos(x) and E_cross=E_in*sin(x), with
    x=pi*(V+V_bias)/(2*V_pi).
    """

    graph_input_ports = ("carrier", "voltage")
    graph_output_ports = ("bar", "cross")

    def __init__(
        self,
        v_pi: float = 1.0,
        bias_voltage: float = 0.0,
        splitting_ratio: float = 0.5,
        insertion_loss_db: float = 0.0,
        *,
        trainable: bool = True,
        coupler_trainable: bool | None = None,
    ) -> None:
        super().__init__()
        if v_pi <= 0:
            raise ValueError("v_pi must be positive")
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        if coupler_trainable is None:
            coupler_trainable = trainable
        self.v_pi = nn.Parameter(torch.tensor(float(v_pi)), requires_grad=trainable)
        self.bias_voltage = nn.Parameter(torch.tensor(float(bias_voltage)), requires_grad=trainable)
        self.splitter = PowerSplitter1x2(splitting_ratio, trainable=coupler_trainable)
        self.output_coupler = DirectionalCoupler2x2(splitting_ratio, trainable=coupler_trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def _half_phase(self, voltage: Tensor, dtype: torch.dtype) -> Tensor:
        _require_real_voltage(voltage)
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(dtype).eps).to(
            device=voltage.device, dtype=dtype
        )
        bias = self.bias_voltage.to(device=voltage.device, dtype=dtype)
        return torch.pi * (voltage.to(dtype=dtype) + bias) / (2.0 * v_pi)

    def forward(self, carrier: Tensor, voltage: Tensor) -> tuple[Tensor, Tensor]:
        require_complex_field(carrier)
        _require_real_voltage(voltage)
        if voltage.numel() != 1 and voltage.shape[-1] != carrier.shape[-1]:
            raise ValueError("carrier and voltage must share their final sample dimension")
        arm0, arm1 = self.splitter(carrier)
        phase = self._half_phase(voltage.to(device=carrier.device), carrier.real.dtype)
        phasor = torch.polar(torch.ones_like(phase), phase)
        output0, output1 = self.output_coupler(arm0 * phasor, arm1 * phasor.conj())
        loss = _amplitude_loss(self.insertion_loss_db, carrier)
        return -1j * loss * output1, -1j * loss * output0

    def field_transmissions(self, voltage: Tensor, *, real_dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        _require_real_voltage(voltage)
        dtype = real_dtype or (voltage.dtype if voltage.is_floating_point() else torch.float32)
        scalar = voltage.ndim == 0
        working_voltage = voltage.reshape(1) if scalar else voltage
        unit = torch.complex(
            torch.ones_like(working_voltage, dtype=dtype),
            torch.zeros_like(working_voltage, dtype=dtype),
        )
        bar, cross = self(unit, working_voltage)
        return (bar.squeeze(0), cross.squeeze(0)) if scalar else (bar, cross)


class FourPortMZM(GraphDevice):
    """Two-input/two-output (four optical port) differential MZM."""

    graph_input_ports = ("input0", "input1", "voltage")
    graph_output_ports = ("output0", "output1")

    def __init__(
        self,
        v_pi: float = 1.0,
        bias_voltage: float = 0.0,
        splitting_ratio: float = 0.5,
        insertion_loss_db: float = 0.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if v_pi <= 0:
            raise ValueError("v_pi must be positive")
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.v_pi = nn.Parameter(torch.tensor(float(v_pi)), requires_grad=trainable)
        self.bias_voltage = nn.Parameter(torch.tensor(float(bias_voltage)), requires_grad=trainable)
        self.input_coupler = DirectionalCoupler2x2(splitting_ratio, trainable=trainable)
        self.output_coupler = DirectionalCoupler2x2(splitting_ratio, trainable=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def forward(self, input0: Tensor, input1: Tensor, voltage: Tensor) -> tuple[Tensor, Tensor]:
        require_complex_field(input0)
        require_complex_field(input1)
        _require_real_voltage(voltage)
        if input0.shape != input1.shape:
            raise ValueError("the two optical inputs must have identical shapes")
        if voltage.numel() != 1 and voltage.shape[-1] != input0.shape[-1]:
            raise ValueError("voltage must be scalar or share the final sample dimension")
        arm0, arm1 = self.input_coupler(input0, input1)
        dtype = input0.real.dtype
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(dtype).eps).to(input0.device, dtype)
        phase = torch.pi * (voltage.to(device=input0.device, dtype=dtype) + self.bias_voltage.to(input0.device, dtype)) / (2.0 * v_pi)
        phasor = torch.polar(torch.ones_like(phase), phase)
        raw0, raw1 = self.output_coupler(arm0 * phasor, arm1 * phasor.conj())
        loss = _amplitude_loss(self.insertion_loss_db, input0)
        return -1j * loss * raw1, -1j * loss * raw0


class IdealMZM(GraphDevice):
    """Ideal single-input/single-output MZM built from the dual-port core."""

    graph_input_ports = ("carrier", "voltage")

    def __init__(self, v_pi: float = 1.0, bias_voltage: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        self.interferometer = DifferentialDualPortMZM(
            v_pi=v_pi,
            bias_voltage=bias_voltage,
            splitting_ratio=0.5,
            trainable=trainable,
            coupler_trainable=False,
        )

    @property
    def v_pi(self) -> Tensor:
        return self.interferometer.v_pi

    @property
    def bias_voltage(self) -> Tensor:
        return self.interferometer.bias_voltage

    def field_transmission(self, voltage: Tensor, *, real_dtype: torch.dtype | None = None) -> Tensor:
        return self.interferometer.field_transmissions(voltage, real_dtype=real_dtype)[0]

    def voltage_for_power_transmission(self, transmission: Tensor) -> Tensor:
        if not isinstance(transmission, Tensor) or torch.is_complex(transmission) or not transmission.is_floating_point():
            raise TypeError("transmission must be a real floating-point torch.Tensor")
        if torch.any((transmission.detach() < 0) | (transmission.detach() > 1)):
            raise ValueError("transmission must lie in [0, 1]")
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(transmission.dtype).eps).to(
            transmission.device, transmission.dtype
        )
        bias = self.bias_voltage.to(transmission.device, transmission.dtype)
        return 2.0 * v_pi * torch.acos(torch.sqrt(transmission)) / torch.pi - bias

    def power_transmission_bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return the minimum and maximum reachable power transmission."""
        if not isinstance(reference, Tensor) or torch.is_complex(reference) or not reference.is_floating_point():
            raise TypeError("reference must be a real floating-point torch.Tensor")
        return torch.zeros((), device=reference.device, dtype=reference.dtype), torch.ones(
            (), device=reference.device, dtype=reference.dtype
        )

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        return self.interferometer(carrier, voltage)[0]


class MachZehnderModulator(GraphDevice):
    """Practical analytic MZM with insertion loss and finite extinction ratio.

    The ideal bar and complementary cross outputs come from IdealMZM's shared
    interferometer.  The cross field supplies the finite-extinction leakage,
    so the practical model does not duplicate the MZI transfer derivation.
    """

    graph_input_ports = ("carrier", "voltage")

    def __init__(
        self,
        v_pi: float = 1.0,
        bias_voltage: float = 0.0,
        insertion_loss_db: float = 0.0,
        extinction_ratio_db: float = 60.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        if extinction_ratio_db < 0:
            raise ValueError("extinction_ratio_db must not be negative")
        self.ideal = IdealMZM(v_pi=v_pi, bias_voltage=bias_voltage, trainable=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)
        self.extinction_ratio_db = nn.Parameter(torch.tensor(float(extinction_ratio_db)), requires_grad=trainable)

    @property
    def v_pi(self) -> Tensor:
        return self.ideal.v_pi

    @property
    def bias_voltage(self) -> Tensor:
        return self.ideal.bias_voltage

    def field_transmission(self, voltage: Tensor, *, real_dtype: torch.dtype | None = None) -> Tensor:
        _require_real_voltage(voltage)
        dtype = real_dtype or (voltage.dtype if voltage.is_floating_point() else torch.float32)
        bar, cross = self.ideal.interferometer.field_transmissions(voltage, real_dtype=dtype)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(voltage.device, dtype)
        er_db = nonnegative_parameter(self.extinction_ratio_db, "extinction_ratio_db").to(voltage.device, dtype)
        ten = torch.tensor(10.0, device=voltage.device, dtype=dtype)
        amplitude_loss = torch.pow(ten, -loss_db / 20.0)
        leakage = torch.sqrt(torch.pow(ten, -er_db / 10.0))
        return amplitude_loss * (bar + 1j * leakage * cross)

    def voltage_for_power_transmission(self, transmission: Tensor) -> Tensor:
        if not isinstance(transmission, Tensor) or torch.is_complex(transmission) or not transmission.is_floating_point():
            raise TypeError("transmission must be a real floating-point torch.Tensor")
        dtype = transmission.dtype
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(transmission.device, dtype)
        er_db = nonnegative_parameter(self.extinction_ratio_db, "extinction_ratio_db").to(transmission.device, dtype)
        ten = torch.tensor(10.0, device=transmission.device, dtype=dtype)
        eta = torch.pow(ten, -loss_db / 10.0)
        epsilon = torch.pow(ten, -er_db / 10.0)
        normalized = (transmission / eta - epsilon) / (1.0 - epsilon)
        if torch.any((normalized.detach() < -1e-6) | (normalized.detach() > 1.0 + 1e-6)):
            raise ValueError("requested transmission is outside this practical MZM's reachable range")
        return self.ideal.voltage_for_power_transmission(normalized.clamp(0.0, 1.0))

    def power_transmission_bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return leakage-floor and peak transmission, including insertion loss."""
        if not isinstance(reference, Tensor) or torch.is_complex(reference) or not reference.is_floating_point():
            raise TypeError("reference must be a real floating-point torch.Tensor")
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(
            reference.device, reference.dtype
        )
        er_db = nonnegative_parameter(self.extinction_ratio_db, "extinction_ratio_db").to(
            reference.device, reference.dtype
        )
        ten = torch.tensor(10.0, device=reference.device, dtype=reference.dtype)
        peak = torch.pow(ten, -loss_db / 10.0)
        return peak * torch.pow(ten, -er_db / 10.0), peak

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(carrier)
        _require_real_voltage(voltage)
        if carrier.shape[-1] != voltage.shape[-1] and voltage.numel() != 1:
            raise ValueError("carrier and voltage must share their final sample dimension")
        return carrier * self.field_transmission(voltage, real_dtype=carrier.real.dtype)


class PracticalMZM(MachZehnderModulator):
    """Explicit name for the practical analytic MZM; legacy name is retained."""


class MeasuredMZM(GraphDevice):
    """MZM driven by a measured voltage-to-power-transmission curve."""

    graph_input_ports = ("carrier", "voltage")

    def __init__(
        self,
        voltage_points_v: Tensor,
        power_transmission: Tensor,
        phase_rad: Tensor | None = None,
        *,
        trainable_curve: bool = False,
    ) -> None:
        super().__init__()
        voltage_points = torch.as_tensor(voltage_points_v, dtype=torch.float32)
        transmission = torch.as_tensor(power_transmission, dtype=torch.float32)
        if voltage_points.ndim != 1 or voltage_points.numel() < 2:
            raise ValueError("voltage_points_v must be one-dimensional with at least two points")
        if transmission.shape != voltage_points.shape:
            raise ValueError("power_transmission must match voltage_points_v")
        if torch.any(torch.diff(voltage_points) <= 0):
            raise ValueError("voltage_points_v must be strictly increasing")
        if torch.any((transmission < 0) | (transmission > 1)):
            raise ValueError("power_transmission must lie in [0, 1]")
        phase = torch.zeros_like(voltage_points) if phase_rad is None else torch.as_tensor(phase_rad, dtype=torch.float32)
        if phase.shape != voltage_points.shape:
            raise ValueError("phase_rad must match voltage_points_v")
        self.register_buffer("voltage_points_v", voltage_points)
        self.power_transmission = nn.Parameter(transmission.clone(), requires_grad=trainable_curve)
        self.phase_rad = nn.Parameter(phase.clone(), requires_grad=trainable_curve)

    @staticmethod
    def _interpolate(x: Tensor, xp: Tensor, yp: Tensor) -> Tensor:
        x_clamped = x.clamp(min=xp[0], max=xp[-1])
        upper = torch.searchsorted(xp, x_clamped, right=True).clamp(1, xp.numel() - 1)
        lower = upper - 1
        fraction = (x_clamped - xp[lower]) / (xp[upper] - xp[lower])
        return yp[lower] + fraction * (yp[upper] - yp[lower])

    def field_transmission(self, voltage: Tensor, *, real_dtype: torch.dtype | None = None) -> Tensor:
        _require_real_voltage(voltage)
        dtype = real_dtype or (voltage.dtype if voltage.is_floating_point() else torch.float32)
        points = self.voltage_points_v.to(device=voltage.device, dtype=dtype)
        transmission = self.power_transmission.to(device=voltage.device, dtype=dtype)
        if torch.any((transmission.detach() < 0) | (transmission.detach() > 1)):
            raise ValueError("trained power_transmission left the physical interval [0, 1]")
        phase_points = self.phase_rad.to(device=voltage.device, dtype=dtype)
        voltage = voltage.to(dtype=dtype)
        power = self._interpolate(voltage, points, transmission)
        phase = self._interpolate(voltage, points, phase_points)
        return torch.polar(torch.sqrt(power.clamp_min(0.0)), phase)

    def voltage_for_power_transmission(self, transmission: Tensor) -> Tensor:
        if not isinstance(transmission, Tensor) or torch.is_complex(transmission) or not transmission.is_floating_point():
            raise TypeError("transmission must be a real floating-point torch.Tensor")
        curve = self.power_transmission.to(device=transmission.device, dtype=transmission.dtype)
        differences = torch.diff(curve.detach())
        ascending = bool(torch.all(differences >= 0))
        descending = bool(torch.all(differences <= 0))
        if not (ascending or descending):
            raise ValueError("inverse voltage lookup requires a monotonic measured transmission curve")
        voltages = self.voltage_points_v.to(device=transmission.device, dtype=transmission.dtype)
        if descending:
            curve = curve.flip(0)
            voltages = voltages.flip(0)
        if torch.any((transmission.detach() < curve[0].detach()) | (transmission.detach() > curve[-1].detach())):
            raise ValueError("requested transmission lies outside the measured curve")
        return self._interpolate(transmission, curve, voltages)

    def power_transmission_bounds(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        """Return the measured curve's reachable power-transmission interval."""
        if not isinstance(reference, Tensor) or torch.is_complex(reference) or not reference.is_floating_point():
            raise TypeError("reference must be a real floating-point torch.Tensor")
        curve = self.power_transmission.to(device=reference.device, dtype=reference.dtype)
        if torch.any((curve.detach() < 0) | (curve.detach() > 1)):
            raise ValueError("trained power_transmission left the physical interval [0, 1]")
        return curve.min(), curve.max()

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(carrier)
        if voltage.numel() != 1 and carrier.shape[-1] != voltage.shape[-1]:
            raise ValueError("carrier and voltage must share their final sample dimension")
        return carrier * self.field_transmission(voltage, real_dtype=carrier.real.dtype)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        voltage_column: str = "voltage_v",
        transmission_column: str = "power_transmission",
        phase_column: str | None = None,
        trainable_curve: bool = False,
    ) -> "MeasuredMZM":
        voltages: list[float] = []
        transmissions: list[float] = []
        phases: list[float] | None = [] if phase_column is not None else None
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {voltage_column, transmission_column}
            if phase_column is not None:
                required.add(phase_column)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV is missing columns: {sorted(missing)}")
            for row in reader:
                voltages.append(float(row[voltage_column]))
                transmissions.append(float(row[transmission_column]))
                if phases is not None and phase_column is not None:
                    phases.append(float(row[phase_column]))
        return cls(
            torch.tensor(voltages),
            torch.tensor(transmissions),
            None if phases is None else torch.tensor(phases),
            trainable_curve=trainable_curve,
        )


class MZMWithElectricalFilter(GraphDevice):
    """Electrical low-pass filter followed by a selected MZM implementation."""

    graph_input_ports = ("carrier", "voltage")

    def __init__(
        self,
        grid: SimulationGrid,
        cutoff_hz: float,
        *,
        order: int = 1,
        filter_trainable: bool = False,
        mzm: GraphDevice | None = None,
    ) -> None:
        super().__init__()
        self.electrical_filter = FrequencyDomainLowPass(grid, cutoff_hz, order, trainable=filter_trainable)
        self.mzm = mzm if mzm is not None else MachZehnderModulator()

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        _require_real_voltage(voltage)
        filtered_voltage = self.electrical_filter(torch.complex(voltage, torch.zeros_like(voltage))).real
        return self.mzm(carrier, filtered_voltage)


class IntensityMZM(GraphDevice):
    """独立的无符号强度 MZI 调制器。

    该模型直接使用常见强度传输函数，而不复用差分双输出 MZM：

    ``T(V)=eta*[epsilon+(1-epsilon)*cos^2(pi*(V+V_bias)/(2*V_pi))]``。

    其中 ``eta`` 是插入损耗对应的峰值功率透射率，``epsilon`` 是有限
    消光比对应的谷值/峰值功率比。输出场取非负平方根，因此本器件只编码
    强度，不承担复权重的符号或相位编码。
    """

    graph_input_ports = ("carrier", "voltage")

    def __init__(
        self,
        v_pi: float = 1.0,
        bias_voltage: float = 0.0,
        insertion_loss_db: float = 0.0,
        extinction_ratio_db: float = 60.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if v_pi <= 0:
            raise ValueError("v_pi must be positive")
        if insertion_loss_db < 0 or extinction_ratio_db < 0:
            raise ValueError("loss and extinction ratio must not be negative")
        self.v_pi = nn.Parameter(torch.tensor(float(v_pi)), requires_grad=trainable)
        self.bias_voltage = nn.Parameter(torch.tensor(float(bias_voltage)), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)
        self.extinction_ratio_db = nn.Parameter(torch.tensor(float(extinction_ratio_db)), requires_grad=trainable)

    def power_transmission(self, voltage: Tensor) -> Tensor:
        _require_real_voltage(voltage)
        dtype = voltage.dtype if voltage.is_floating_point() else torch.float32
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(dtype).eps).to(
            voltage.device, dtype
        )
        bias = self.bias_voltage.to(voltage.device, dtype)
        loss = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(voltage.device, dtype)
        extinction = nonnegative_parameter(self.extinction_ratio_db, "extinction_ratio_db").to(
            voltage.device, dtype
        )
        ten = torch.tensor(10.0, device=voltage.device, dtype=dtype)
        eta = torch.pow(ten, -loss / 10.0)
        epsilon = torch.pow(ten, -extinction / 10.0)
        phase = torch.pi * (voltage.to(dtype=dtype) + bias) / (2.0 * v_pi)
        return eta * (epsilon + (1.0 - epsilon) * torch.cos(phase).square())

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(carrier)
        if voltage.numel() != 1 and voltage.shape[-1] != carrier.shape[-1]:
            raise ValueError("carrier and voltage must share their final sample dimension")
        return carrier * torch.sqrt(self.power_transmission(voltage).to(carrier.real.dtype))


class IntensityMZMActivation(nn.Module):
    """把神经网络实值张量映射到强度 MZI 的非线性传输曲线。

    ``input_scale_v_per_unit`` 把无量纲网络值换算为驱动电压。默认偏置位于
    四分之一波电压附近，使零点处处于斜率较大的非线性区间。输出是无量纲
    功率透射率，可替换 ReLU 等电子激活函数做对照实验。
    """

    def __init__(
        self,
        *,
        v_pi: float = 1.0,
        bias_voltage: float = -0.5,
        input_scale_v_per_unit: float = 0.25,
        insertion_loss_db: float = 0.0,
        extinction_ratio_db: float = 30.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if input_scale_v_per_unit < 0:
            raise ValueError("input_scale_v_per_unit must not be negative")
        self.input_scale_v_per_unit = nn.Parameter(
            torch.tensor(float(input_scale_v_per_unit)), requires_grad=trainable
        )
        self.modulator = IntensityMZM(
            v_pi=v_pi,
            bias_voltage=bias_voltage,
            insertion_loss_db=insertion_loss_db,
            extinction_ratio_db=extinction_ratio_db,
            trainable=trainable,
        )

    def forward(self, values: Tensor) -> Tensor:
        if not isinstance(values, Tensor) or torch.is_complex(values) or not values.is_floating_point():
            raise TypeError("values must be a real floating-point torch.Tensor")
        scale = nonnegative_parameter(self.input_scale_v_per_unit, "input_scale_v_per_unit").to(
            values.device, values.dtype
        )
        return self.modulator.power_transmission(scale * values)


class MZIComplexWeightBank(GraphDevice):
    """用 MZI 内外相移参数化复权重并完成相干乘加。

    每个输出行先按二范数 ``s`` 标定，再使用
    ``w/s=cos(theta/2)*exp(j*phi)``。归一化行可嵌入无源酉 MZI 网络，
    ``s`` 是读出后的已知标定系数。``theta`` 对应 MZI 内部相移，``phi``
    对应外部相移；这与 Nature Communications 12, 457 (2021) 的复值
    MZI 权重参数化一致。输入窗口形状应为
    ``(batch, in_channels, positions, kernel_size)``。
    """

    def __init__(
        self,
        complex_weights: Tensor,
        *,
        insertion_loss_db: float = 0.0,
        phase_error_std_rad: float = 0.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(complex_weights)
        if not torch.is_complex(weights) or weights.ndim != 3:
            raise ValueError("complex_weights must have shape (out_channels, in_channels, kernel_size)")
        if insertion_loss_db < 0 or phase_error_std_rad < 0:
            raise ValueError("insertion loss and phase error must not be negative")
        real_dtype = weights.real.dtype
        row_norm = torch.linalg.vector_norm(weights, dim=(1, 2), keepdim=True)
        scale = torch.where(row_norm > 0.0, row_norm, torch.ones_like(row_norm))
        normalized_magnitude = (weights.abs() / scale).clamp(0.0, 1.0)
        theta = 2.0 * torch.acos(normalized_magnitude)
        phi = torch.angle(weights)
        self.theta_rad = nn.Parameter(theta.to(dtype=real_dtype), requires_grad=trainable)
        self.phi_rad = nn.Parameter(phi.to(dtype=real_dtype), requires_grad=trainable)
        self.register_buffer("weight_scale", scale.to(dtype=real_dtype))
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)
        self.phase_error_std_rad = nn.Parameter(torch.tensor(float(phase_error_std_rad)), requires_grad=False)

    @property
    def out_channels(self) -> int:
        return self.theta_rad.shape[0]

    @property
    def in_channels(self) -> int:
        return self.theta_rad.shape[1]

    @property
    def kernel_size(self) -> int:
        return self.theta_rad.shape[2]

    def effective_weights(self) -> Tensor:
        dtype = self.theta_rad.dtype
        device = self.theta_rad.device
        loss = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device, dtype)
        phase_error = nonnegative_parameter(self.phase_error_std_rad, "phase_error_std_rad").to(device, dtype)
        phase = self.phi_rad
        if float(phase_error.detach()) > 0.0:
            phase = phase + torch.randn_like(phase) * phase_error
        ten = torch.tensor(10.0, device=device, dtype=dtype)
        field_loss = torch.pow(ten, -loss / 20.0)
        amplitude = self.weight_scale.to(device, dtype) * field_loss * torch.cos(self.theta_rad / 2.0)
        return torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))

    def forward(self, windows: Tensor) -> Tensor:
        require_complex_field(windows)
        if windows.ndim != 4:
            raise ValueError("windows must have shape (batch, in_channels, positions, kernel_size)")
        if windows.shape[1] != self.in_channels or windows.shape[-1] != self.kernel_size:
            raise ValueError("window channels/kernel size do not match the MZI weight bank")
        weights = self.effective_weights().to(device=windows.device, dtype=windows.dtype)
        return torch.einsum("nclk,ock->nol", windows, weights)
