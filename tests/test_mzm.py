import math

import torch

from optosimlab.devices.mzm import MachZehnderModulator


def complex_ones(samples: int = 32) -> torch.Tensor:
    return torch.ones(samples, dtype=torch.complex64)


def test_ideal_mzm_has_peak_and_null_at_expected_voltages() -> None:
    mzm = MachZehnderModulator(v_pi=2.0, extinction_ratio_db=120.0, trainable=False)
    carrier = complex_ones()
    peak = mzm(carrier, torch.zeros(32))
    null = mzm(carrier, torch.full((32,), 2.0))
    assert torch.allclose(peak, carrier, atol=1e-6)
    assert torch.max(null.abs()) < 2e-6


def test_mzm_loss_and_extinction_are_power_correct() -> None:
    mzm = MachZehnderModulator(v_pi=1.0, insertion_loss_db=3.0, extinction_ratio_db=20.0, trainable=False)
    carrier = complex_ones()
    peak_power = mzm(carrier, torch.zeros(32)).abs().square().mean()
    null_power = mzm(carrier, torch.ones(32)).abs().square().mean()
    assert torch.allclose(peak_power, torch.tensor(10 ** (-3.0 / 10.0)), rtol=1e-5)
    assert torch.allclose(null_power / peak_power, torch.tensor(10 ** (-20.0 / 10.0)), rtol=1e-4)


def test_mzm_parameters_are_registered_and_differentiable() -> None:
    mzm = MachZehnderModulator(v_pi=1.0, bias_voltage=0.2)
    names = set(dict(mzm.named_parameters()))
    assert names == {"v_pi", "bias_voltage", "insertion_loss_db", "extinction_ratio_db"}
    output_power = mzm(complex_ones(), torch.zeros(32)).abs().square().mean()
    output_power.backward()
    assert mzm.bias_voltage.grad is not None
    assert not math.isclose(mzm.bias_voltage.grad.item(), 0.0, abs_tol=1e-7)
