"""Mach-Zehnder electro-optic modulation models."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field
from .filters import FrequencyDomainLowPass


class MeasuredMZM(nn.Module):
    """MZM driven by a measured voltage-to-transmission calibration curve.

    Voltage is in V, power_transmission is dimensionless, and optional phase
    is in rad.  Values are linearly interpolated and clamped to the first/last
    measurement outside the calibrated voltage range.  Optical field amplitude
    is the square root of the measured power transmission.
    """

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
        if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
            raise TypeError("voltage must be a real-valued torch.Tensor")
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
        """Load a calibration curve from a CSV file with named columns."""
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


class MachZehnderModulator(nn.Module):
    """Memoryless push-pull MZM operating on a complex optical envelope.

    For voltage ``V`` and bias ``Vb``, the ideal field model is
    ``E_out = E_in cos(pi (V + Vb) / (2 Vpi))``.  With finite extinction
    ratio ``ER`` and power insertion transmission ``eta`` it becomes::

        E_out = sqrt(eta) E_in [cos(x) + j sqrt(10 ** (-ER / 10)) sin(x)]

    Thus peak power is ``eta |E_in|^2`` and null power divided by peak power
    is exactly ``10 ** (-ER / 10)``.  All four adjustable physical values are
    registered ``nn.Parameter`` instances and may be optimised by PyTorch.
    """

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
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        if extinction_ratio_db < 0:
            raise ValueError("extinction_ratio_db must not be negative")
        self.v_pi = nn.Parameter(torch.tensor(float(v_pi)), requires_grad=trainable)
        self.bias_voltage = nn.Parameter(torch.tensor(float(bias_voltage)), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)
        self.extinction_ratio_db = nn.Parameter(torch.tensor(float(extinction_ratio_db)), requires_grad=trainable)

    def field_transmission(self, voltage: Tensor, *, real_dtype: torch.dtype | None = None) -> Tensor:
        """Return the complex field transfer coefficient for a real drive voltage."""
        if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
            raise TypeError("voltage must be a real-valued torch.Tensor")
        dtype = real_dtype or (voltage.dtype if torch.is_floating_point(voltage) else torch.float32)
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(dtype).eps).to(device=voltage.device, dtype=dtype)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=voltage.device, dtype=dtype)
        er_db = nonnegative_parameter(self.extinction_ratio_db, "extinction_ratio_db").to(device=voltage.device, dtype=dtype)
        bias = self.bias_voltage.to(device=voltage.device, dtype=dtype)
        x = torch.pi * (voltage.to(dtype=dtype) + bias) / (2.0 * v_pi)
        ten = torch.tensor(10.0, device=voltage.device, dtype=dtype)
        amplitude_loss = torch.pow(ten, -loss_db / 20.0)
        leakage = torch.sqrt(torch.pow(ten, -er_db / 10.0))
        return torch.complex(amplitude_loss * torch.cos(x), amplitude_loss * leakage * torch.sin(x))

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(carrier)
        if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
            raise TypeError("voltage must be a real-valued torch.Tensor")
        if carrier.shape[-1] != voltage.shape[-1] and voltage.numel() != 1:
            raise ValueError("carrier and voltage must share their final sample dimension")
        return carrier * self.field_transmission(voltage, real_dtype=carrier.real.dtype)


class MZMWithElectricalFilter(nn.Module):
    """An electrical low-pass filter followed by a memoryless MZM.

    This corresponds to the repository requirement to compose an ideal MZM and
    an electrical bandwidth limit into a practical electro-optic modulator.
    """

    def __init__(
        self,
        grid: SimulationGrid,
        cutoff_hz: float,
        *,
        order: int = 1,
        filter_trainable: bool = False,
        mzm: MachZehnderModulator | None = None,
    ) -> None:
        super().__init__()
        self.electrical_filter = FrequencyDomainLowPass(grid, cutoff_hz, order, trainable=filter_trainable)
        self.mzm = mzm if mzm is not None else MachZehnderModulator()

    def forward(self, carrier: Tensor, voltage: Tensor) -> Tensor:
        if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
            raise TypeError("voltage must be a real-valued torch.Tensor")
        filtered_voltage = self.electrical_filter(torch.complex(voltage, torch.zeros_like(voltage))).real
        return self.mzm(carrier, filtered_voltage)
