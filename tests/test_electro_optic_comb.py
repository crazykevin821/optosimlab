import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.electro_optic_comb import ElectroOpticComb


def test_eo_comb_zero_modulation_is_identity() -> None:
    module = ElectroOpticComb(SimulationGrid(64.0), 8.0, modulation_indices_rad=0.0, trainable=False)
    field = torch.randn(3, 64, dtype=torch.complex64)
    assert torch.allclose(module(field), field, atol=1e-6)


def test_eo_comb_matches_phase_modulation_formula_at_every_sample() -> None:
    module = ElectroOpticComb(
        SimulationGrid(64.0),
        modulation_frequency_hz=torch.tensor([4.0, 8.0]),
        modulation_indices_rad=torch.tensor([0.2, 0.5]),
        rf_phases_rad=torch.tensor([0.1, -0.3]),
        trainable=False,
    )
    samples = 64
    t = torch.arange(samples) / 64.0
    expected_phase = 0.2 * torch.sin(2 * torch.pi * 4.0 * t + 0.1) + 0.5 * torch.sin(2 * torch.pi * 8.0 * t - 0.3)
    expected = torch.exp(1j * expected_phase)
    assert torch.allclose(module(torch.ones(samples, dtype=torch.complex64)), expected, atol=2e-6)


def test_eo_comb_has_bessel_carrier_and_first_sidebands() -> None:
    beta = 0.7
    module = ElectroOpticComb(SimulationGrid(128.0), 4.0, modulation_indices_rad=beta, trainable=False)
    spectrum = torch.fft.fft(module(torch.ones(128, dtype=torch.complex64))) / 128.0
    expected_j0 = torch.special.bessel_j0(torch.tensor(beta))
    expected_j1 = torch.special.bessel_j1(torch.tensor(beta))
    assert torch.allclose(spectrum[0].real, expected_j0, atol=2e-6)
    assert torch.allclose(spectrum[4].real, expected_j1, atol=2e-6)
    assert torch.allclose(spectrum[-4].real, -expected_j1, atol=2e-6)


def test_eo_comb_preserves_power_before_and_applies_power_db_loss_after() -> None:
    field = torch.randn(128, dtype=torch.complex64)
    ideal = ElectroOpticComb(SimulationGrid(128.0), 8.0, modulation_indices_rad=1.3, trainable=False)
    lossy = ElectroOpticComb(SimulationGrid(128.0), 8.0, modulation_indices_rad=1.3, insertion_loss_db=3.0, trainable=False)
    input_power = field.abs().square().mean()
    assert torch.allclose(ideal(field).abs().square().mean(), input_power, rtol=2e-6, atol=2e-6)
    assert torch.allclose(lossy(field).abs().square().mean(), input_power * 10 ** (-3.0 / 10.0), rtol=2e-6, atol=2e-6)


def test_eo_comb_trainable_parameters_receive_gradients() -> None:
    module = ElectroOpticComb(SimulationGrid(64.0), 8.0, modulation_indices_rad=0.4, rf_phases_rad=0.2, insertion_loss_db=1.0)
    loss = module(torch.ones(64, dtype=torch.complex64)).real.mean()
    loss.backward()
    assert set(dict(module.named_parameters())) == {"modulation_indices_rad", "rf_phases_rad", "insertion_loss_db"}
    assert module.modulation_indices_rad.grad is not None
    assert module.rf_phases_rad.grad is not None
    assert module.insertion_loss_db.grad is not None
    assert torch.all(module.modulation_indices_rad.grad.abs() > 1e-7)


def test_eo_comb_rejects_invalid_input_and_nonperiodic_or_nyquist_drive() -> None:
    with pytest.raises(TypeError, match="complex"):
        ElectroOpticComb(SimulationGrid(64.0), 8.0)(torch.ones(64))
    with pytest.raises(ValueError, match="integer multiple"):
        ElectroOpticComb(SimulationGrid(64.0), 7.5)(torch.ones(64, dtype=torch.complex64))
    with pytest.raises(ValueError, match="Nyquist"):
        ElectroOpticComb(SimulationGrid(64.0), 32.0)(torch.ones(64, dtype=torch.complex64))
