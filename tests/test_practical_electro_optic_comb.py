import math

import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.electro_optic_comb import ElectroOpticComb
from optosimlab.devices.practical_electro_optic_comb import PracticalElectroOpticComb


def test_practical_eo_comb_recovers_ideal_model_with_large_bandwidth_and_no_noise() -> None:
    grid = SimulationGrid(64.0)
    ideal = ElectroOpticComb(grid, 8.0, modulation_indices_rad=0.7, rf_phases_rad=0.2, trainable=False)
    practical = PracticalElectroOpticComb(
        grid,
        8.0,
        modulation_indices_rad=0.7,
        rf_phases_rad=0.2,
        rf_bandwidth_hz=1e12,
        enable_phase_noise=False,
        trainable=False,
    )
    field = torch.randn(64, dtype=torch.complex64)
    assert torch.allclose(practical(field), ideal(field), atol=2e-6, rtol=2e-6)


def test_first_order_rf_response_has_expected_amplitude_and_phase_lag() -> None:
    grid = SimulationGrid(64.0)
    module = PracticalElectroOpticComb(
        grid,
        8.0,
        modulation_indices_rad=1.0,
        rf_bandwidth_hz=8.0,
        enable_phase_noise=False,
        trainable=False,
    )
    t = torch.arange(64) / grid.sample_rate_hz
    expected_phase = (1.0 / math.sqrt(2.0)) * torch.sin(2 * torch.pi * 8.0 * t - torch.pi / 4)
    expected = torch.exp(1j * expected_phase)
    assert torch.allclose(module(torch.ones(64, dtype=torch.complex64)), expected, atol=2e-6)


def test_phase_noise_changes_phase_but_preserves_power_and_can_be_seeded() -> None:
    grid = SimulationGrid(64.0)
    noisy = PracticalElectroOpticComb(
        grid,
        8.0,
        modulation_indices_rad=0.8,
        rf_bandwidth_hz=1e12,
        rf_phase_noise_diffusion_rad2_per_s=16.0,
        trainable=False,
    )
    noiseless = PracticalElectroOpticComb(
        grid,
        8.0,
        modulation_indices_rad=0.8,
        rf_bandwidth_hz=1e12,
        enable_phase_noise=False,
        trainable=False,
    )
    field = torch.ones(64, dtype=torch.complex64)
    first = noisy(field, generator=torch.Generator().manual_seed(7))
    repeat = noisy(field, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(first, repeat)
    assert not torch.allclose(first, noiseless(field))
    assert torch.allclose(first.abs().square().mean(), field.abs().square().mean(), atol=2e-6)


def test_practical_eo_comb_parameters_are_trainable() -> None:
    module = PracticalElectroOpticComb(
        SimulationGrid(64.0),
        8.0,
        modulation_indices_rad=0.7,
        rf_phases_rad=0.2,
        insertion_loss_db=1.0,
        rf_bandwidth_hz=16.0,
        rf_phase_noise_diffusion_rad2_per_s=16.0,
    )
    output = module(torch.ones(64, dtype=torch.complex64), generator=torch.Generator().manual_seed(3))
    output.real.mean().backward()
    assert set(dict(module.named_parameters())) == {
        "modulation_indices_rad",
        "rf_phases_rad",
        "insertion_loss_db",
        "rf_bandwidth_hz",
        "rf_phase_noise_diffusion_rad2_per_s",
    }
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().item() > 1e-7


def test_practical_eo_comb_rejects_invalid_rf_parameters_and_real_fields() -> None:
    with pytest.raises(ValueError, match="rf_bandwidth_hz"):
        PracticalElectroOpticComb(SimulationGrid(64.0), 8.0, rf_bandwidth_hz=0.0)
    with pytest.raises(ValueError, match="rf_phase_noise_diffusion"):
        PracticalElectroOpticComb(SimulationGrid(64.0), 8.0, rf_phase_noise_diffusion_rad2_per_s=-1.0)
    module = PracticalElectroOpticComb(SimulationGrid(64.0), 8.0)
    with pytest.raises(TypeError, match="complex"):
        module(torch.ones(64))
    with torch.no_grad():
        module.rf_bandwidth_hz.fill_(-1.0)
    with pytest.raises(ValueError, match="rf_bandwidth_hz"):
        module(torch.ones(64, dtype=torch.complex64))
