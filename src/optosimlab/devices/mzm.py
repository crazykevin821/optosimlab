"""Mach-Zehnder electro-optic modulation models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from .base import nonnegative_parameter, require_complex_field
from .filters import FrequencyDomainLowPass


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
