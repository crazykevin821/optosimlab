import torch

from optosimlab import IntensityMZM, IntensityMZMActivation


def test_unsigned_intensity_mzm_matches_direct_power_formula() -> None:
    mzm = IntensityMZM(v_pi=2.0, bias_voltage=0.0, trainable=False)
    voltage = torch.tensor([0.0, 1.0, 2.0])
    epsilon = 10.0 ** (-60.0 / 10.0)
    expected = epsilon + (1.0 - epsilon) * torch.cos(torch.pi * voltage / 4.0).square()
    assert torch.allclose(mzm.power_transmission(voltage), expected, atol=1e-6)
    output = mzm(torch.ones(3, dtype=torch.complex64), voltage)
    assert torch.allclose(output.abs().square(), expected, atol=1e-6)


def test_intensity_mzm_activation_is_nonnegative_and_trainable() -> None:
    activation = IntensityMZMActivation()
    values = torch.linspace(-1.0, 1.0, 17, requires_grad=True)
    result = activation(values)
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)
    result.sum().backward()
    assert values.grad is not None and torch.all(torch.isfinite(values.grad))
    assert activation.input_scale_v_per_unit.grad is not None
    assert activation.modulator.bias_voltage.grad is not None
