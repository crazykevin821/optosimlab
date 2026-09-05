"""Loss-aware splitters, couplers and Mach-Zehnder interferometers."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import nonnegative_parameter, require_complex_field


def _ratio_logit(splitting_ratio: float) -> Tensor:
    if not 0.0 < splitting_ratio < 1.0:
        raise ValueError("splitting_ratio must be strictly between 0 and 1")
    ratio = torch.tensor(float(splitting_ratio))
    return torch.logit(ratio)


def _amplitude_loss(loss_db: Tensor, reference: Tensor) -> Tensor:
    checked = nonnegative_parameter(loss_db, "insertion_loss_db").to(
        device=reference.device, dtype=reference.real.dtype
    )
    return torch.pow(torch.tensor(10.0, device=reference.device, dtype=reference.real.dtype), -checked / 20.0)


class PowerSplitter1x2(nn.Module):
    """One-input/two-output loss-aware splitter with trainable power ratio.

    For power ratio r the two fields are sqrt(r) E and sqrt(1-r) E.
    Therefore their total output power equals the input power times the stated
    insertion-loss transmission.
    """

    def __init__(self, splitting_ratio: float = 0.5, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.splitting_logit = nn.Parameter(_ratio_logit(splitting_ratio), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    @property
    def splitting_ratio(self) -> Tensor:
        """Power fraction delivered to output port 0."""
        return torch.sigmoid(self.splitting_logit)

    def forward(self, field: Tensor) -> tuple[Tensor, Tensor]:
        require_complex_field(field)
        ratio = self.splitting_ratio.to(device=field.device, dtype=field.real.dtype)
        loss = _amplitude_loss(self.insertion_loss_db, field)
        return loss * torch.sqrt(ratio) * field, loss * torch.sqrt(1.0 - ratio) * field


class DirectionalCoupler2x2(nn.Module):
    """Reciprocal loss-aware 2x2 directional coupler.

    Its field matrix is [[sqrt(r), j*sqrt(1-r)],
    [j*sqrt(1-r), sqrt(r)]].  With zero insertion loss this matrix is unitary;
    the j terms explicitly retain the quadrature phase of cross coupling.
    """

    def __init__(self, splitting_ratio: float = 0.5, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        self.splitting_logit = nn.Parameter(_ratio_logit(splitting_ratio), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    @property
    def splitting_ratio(self) -> Tensor:
        return torch.sigmoid(self.splitting_logit)

    def forward(self, port0: Tensor, port1: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if port1 is None:
            require_complex_field(port0)
            if port0.ndim < 2 or port0.shape[-2] != 2:
                raise ValueError("a stacked coupler input must have shape (..., 2, samples)")
            port0, port1 = port0.unbind(dim=-2)
        require_complex_field(port0)
        require_complex_field(port1)
        if port0.shape != port1.shape:
            raise ValueError("the two coupler input ports must have identical shapes")
        ratio = self.splitting_ratio.to(device=port0.device, dtype=port0.real.dtype)
        through = torch.sqrt(ratio)
        cross = torch.sqrt(1.0 - ratio)
        loss = _amplitude_loss(self.insertion_loss_db, port0)
        return loss * (through * port0 + 1j * cross * port1), loss * (1j * cross * port0 + through * port1)


class DualInputSingleOutputMZI(nn.Module):
    """Two-input/one-observed-output MZI with differential phase drive.

    The two couplers include their cross-port quadrature phases.  At 50:50,
    zero loss and output_port=1, the result is
    j[cos(x) E0 - sin(x) E1], where x=pi(V+bias)/(2 Vpi).
    This differential response supports positive/negative optical branches.
    """

    def __init__(
        self,
        v_pi: float = 1.0,
        bias_voltage: float = 0.0,
        splitting_ratio: float = 0.5,
        insertion_loss_db: float = 0.0,
        *,
        output_port: int = 1,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if v_pi <= 0:
            raise ValueError("v_pi must be positive")
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        if output_port not in (0, 1):
            raise ValueError("output_port must be 0 or 1")
        self.v_pi = nn.Parameter(torch.tensor(float(v_pi)), requires_grad=trainable)
        self.bias_voltage = nn.Parameter(torch.tensor(float(bias_voltage)), requires_grad=trainable)
        self.input_coupler = DirectionalCoupler2x2(splitting_ratio, trainable=trainable)
        self.output_coupler = DirectionalCoupler2x2(splitting_ratio, trainable=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)
        self.output_port = output_port

    def forward(self, input0: Tensor, input1: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(input0)
        require_complex_field(input1)
        if input0.shape != input1.shape:
            raise ValueError("the two MZI optical inputs must have identical shapes")
        if not isinstance(voltage, Tensor) or torch.is_complex(voltage):
            raise TypeError("voltage must be a real-valued torch.Tensor")
        if voltage.numel() != 1 and voltage.shape[-1] != input0.shape[-1]:
            raise ValueError("voltage must be scalar or share the final sample dimension")
        arm0, arm1 = self.input_coupler(input0, input1)
        dtype = input0.real.dtype
        v_pi = nonnegative_parameter(self.v_pi, "v_pi", floor=torch.finfo(dtype).eps).to(input0.device, dtype)
        bias = self.bias_voltage.to(input0.device, dtype)
        half_phase = torch.pi * (voltage.to(device=input0.device, dtype=dtype) + bias) / (2.0 * v_pi)
        arm0 = arm0 * torch.polar(torch.ones_like(half_phase), half_phase)
        arm1 = arm1 * torch.polar(torch.ones_like(half_phase), -half_phase)
        outputs = self.output_coupler(arm0, arm1)
        return _amplitude_loss(self.insertion_loss_db, input0) * outputs[self.output_port]


class SingleInputMZI(DualInputSingleOutputMZI):
    """Single-input/single-observed-output MZI."""

    def forward(self, field: Tensor, voltage: Tensor) -> Tensor:
        require_complex_field(field)
        return super().forward(field, torch.zeros_like(field), voltage)


class FourInputMZI(nn.Module):
    """Legacy four-input coherent combiner retained for v0.7 compatibility.

    Input shape is ``(..., 4, samples)``. The observed output port is
    ``Eout = sqrt(eta) / 2 * sum_k(Ek exp(j phi_k))``. The factor 1/2 is the
    unitary four-way-combiner normalization: one active input yields one fourth
    of its power in this output port; four equal in-phase inputs yield all four
    input powers at this port. Other physical output ports are not represented.
    """

    def __init__(self, phase_rad: Tensor | None = None, insertion_loss_db: float = 0.0, *, trainable: bool = True) -> None:
        super().__init__()
        if insertion_loss_db < 0:
            raise ValueError("insertion_loss_db must not be negative")
        initial_phase = torch.zeros(4) if phase_rad is None else torch.as_tensor(phase_rad, dtype=torch.float32)
        if initial_phase.shape != (4,):
            raise ValueError("phase_rad must have shape (4,)")
        self.phase_rad = nn.Parameter(initial_phase.clone(), requires_grad=trainable)
        self.insertion_loss_db = nn.Parameter(torch.tensor(float(insertion_loss_db)), requires_grad=trainable)

    def forward(self, fields: Tensor) -> Tensor:
        require_complex_field(fields)
        if fields.ndim < 2 or fields.shape[-2] != 4:
            raise ValueError("fields must have shape (..., 4, samples)")
        phase = self.phase_rad.to(device=fields.device, dtype=fields.real.dtype)
        loss_db = nonnegative_parameter(self.insertion_loss_db, "insertion_loss_db").to(device=fields.device, dtype=fields.real.dtype)
        coefficients = torch.polar(torch.ones_like(phase), phase).view(*([1] * (fields.ndim - 2)), 4, 1)
        ten = torch.tensor(10.0, device=fields.device, dtype=fields.real.dtype)
        amplitude = torch.pow(ten, -loss_db / 20.0)
        return amplitude * (fields * coefficients).sum(dim=-2) / 2.0
