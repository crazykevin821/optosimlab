import torch

from optosimlab import DirectionalCoupler2x2, DualInputSingleOutputMZI, PowerSplitter1x2, SingleInputMZI


def test_splitter_conserves_power_and_ratio_is_trainable() -> None:
    splitter = PowerSplitter1x2(0.25)
    field = torch.ones(32, dtype=torch.complex64)
    port0, port1 = splitter(field)
    assert torch.allclose(port0.abs().square() + port1.abs().square(), field.abs().square())
    port0.abs().square().mean().backward()
    assert splitter.splitting_logit.grad is not None


def test_directional_coupler_is_unitary_and_cross_port_has_quadrature_phase() -> None:
    coupler = DirectionalCoupler2x2(0.5, trainable=False)
    field = torch.ones(16, dtype=torch.complex64)
    zero = torch.zeros_like(field)
    port0, port1 = coupler(field, zero)
    assert torch.allclose(port0, torch.full_like(field, 2**-0.5))
    assert torch.allclose(port1, torch.full_like(field, 1j * 2**-0.5))
    assert torch.allclose(port0.abs().square() + port1.abs().square(), field.abs().square())


def test_single_input_mzi_has_analytic_peak_and_null() -> None:
    mzi = SingleInputMZI(v_pi=2.0, trainable=False)
    field = torch.ones(16, dtype=torch.complex64)
    assert torch.allclose(mzi(field, torch.tensor(0.0)), 1j * field, atol=2e-6)
    assert mzi(field, torch.tensor(2.0)).abs().max() < 2e-6


def test_dual_input_mzi_has_differential_response() -> None:
    mzi = DualInputSingleOutputMZI(v_pi=2.0, trainable=False)
    field = torch.ones(16, dtype=torch.complex64)
    zero = torch.zeros_like(field)
    assert torch.allclose(mzi(field, zero, torch.tensor(0.0)), 1j * field, atol=2e-6)
    assert torch.allclose(mzi(zero, field, torch.tensor(2.0)), -1j * field, atol=2e-6)
