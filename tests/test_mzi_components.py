import torch

from optosimlab import (
    DifferentialDualPortMZM,
    DirectionalCoupler2x2,
    FourPortMZM,
    PowerSplitter1x2,
)


def test_splitter_conserves_power_and_cross_port_is_in_quadrature() -> None:
    splitter = PowerSplitter1x2(0.25)
    field = torch.ones(32, dtype=torch.complex64)
    port0, port1 = splitter(field)
    assert torch.allclose(port0, torch.full_like(field, 0.5))
    assert torch.allclose(port1, torch.full_like(field, 1j * (0.75**0.5)))
    assert torch.allclose(port0.abs().square() + port1.abs().square(), field.abs().square())
    port0.abs().square().mean().backward()
    assert splitter.splitting_logit.grad is not None


def test_directional_coupler_is_unitary() -> None:
    coupler = DirectionalCoupler2x2(0.37, trainable=False)
    torch.manual_seed(9)
    input0 = torch.complex(torch.randn(16), torch.randn(16))
    input1 = torch.complex(torch.randn(16), torch.randn(16))
    output0, output1 = coupler(input0, input1)
    assert torch.allclose(
        output0.abs().square() + output1.abs().square(),
        input0.abs().square() + input1.abs().square(),
        rtol=2e-6,
        atol=2e-6,
    )


def test_differential_dual_port_mzm_has_complementary_analytic_outputs() -> None:
    mzi = DifferentialDualPortMZM(v_pi=2.0, trainable=False)
    field = torch.ones(16, dtype=torch.complex64)
    voltage = torch.tensor([0.0, 1.0, 2.0])
    bar, cross = mzi.field_transmissions(voltage)
    angle = torch.pi * voltage / 4.0
    assert torch.allclose(bar, torch.cos(angle).to(torch.complex64), atol=2e-6)
    assert torch.allclose(cross, torch.sin(angle).to(torch.complex64), atol=2e-6)
    port0, port1 = mzi(field, torch.zeros(16))
    assert torch.allclose(port0, field, atol=2e-6)
    assert port1.abs().max() < 2e-6


def test_four_port_mzm_is_composed_from_two_couplers() -> None:
    mzi = FourPortMZM(trainable=False)
    assert isinstance(mzi.input_coupler, DirectionalCoupler2x2)
    assert isinstance(mzi.output_coupler, DirectionalCoupler2x2)
