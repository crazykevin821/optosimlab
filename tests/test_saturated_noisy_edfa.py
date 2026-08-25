import math

import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.edfa import SaturatedNoisyEDFA


def test_saturated_edfa_matches_phenomenological_gain_formula() -> None:
    amplifier = SaturatedNoisyEDFA(
        SimulationGrid(128.0),
        pump_current_ma=60.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=128.0,
        saturation_power=2.0,
        enable_ase=False,
        trainable=False,
    )
    field = torch.ones(128, dtype=torch.complex64) * math.sqrt(2.0)
    expected_power_gain = 10 ** (20.0 / 10.0) / (1.0 + 2.0 / 2.0)
    assert torch.allclose(amplifier(field).abs().square().mean() / field.abs().square().mean(), torch.tensor(expected_power_gain), rtol=2e-5)


def test_saturation_reduces_gain_monotonically_and_low_power_recovers_small_signal() -> None:
    amplifier = SaturatedNoisyEDFA(
        SimulationGrid(128.0),
        pump_current_ma=60.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=128.0,
        saturation_power=1.0,
        enable_ase=False,
        trainable=False,
    )
    small = amplifier.effective_power_gain(128, 1e-6)[0]
    large = amplifier.effective_power_gain(128, 10.0)[0]
    small_signal = amplifier.power_gain(128)[0]
    assert torch.allclose(small, small_signal, rtol=2e-6)
    assert large < small


def test_ase_fft_normalization_matches_integrated_psd_statistically() -> None:
    torch.manual_seed(1234)
    grid = SimulationGrid(256.0)
    samples = 8192
    amplifier = SaturatedNoisyEDFA(
        grid,
        pump_current_ma=40.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=128.0,
        noise_figure_db=5.0,
        trainable=False,
    )
    psd = amplifier.ase_power_spectral_density(samples)
    expected_noise_power = psd.sum() * (grid.sample_rate_hz / samples)
    noise = amplifier(torch.zeros(samples, dtype=torch.complex64))
    assert torch.allclose(noise.abs().square().mean(), expected_noise_power, rtol=0.12, atol=1e-30)


def test_ase_is_zero_when_effective_power_gain_is_not_above_unity() -> None:
    amplifier = SaturatedNoisyEDFA(
        SimulationGrid(128.0),
        pump_current_ma=0.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=1.0,
        gain_bandwidth_hz=128.0,
        trainable=False,
    )
    assert torch.count_nonzero(amplifier.ase_power_spectral_density(128)) == 0
    assert torch.count_nonzero(amplifier(torch.zeros(128, dtype=torch.complex64))) == 0


def test_saturated_noisy_edfa_physical_parameters_are_trainable() -> None:
    amplifier = SaturatedNoisyEDFA(
        SimulationGrid(128.0),
        pump_current_ma=60.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=32.0,
        gain_center_offset_hz=8.0,
        saturation_power=2.0,
        noise_figure_db=5.0,
    )
    loss = amplifier.ase_power_spectral_density(128, signal_power=torch.tensor(1.0)).sum()
    loss.backward()
    assert set(dict(amplifier.named_parameters())) == {
        "pump_current_ma",
        "transparency_current_ma",
        "gain_slope_db_per_ma",
        "gain_bandwidth_hz",
        "gain_center_offset_hz",
        "insertion_loss_db",
        "saturation_power",
        "noise_figure_db",
    }
    for parameter in amplifier.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().item() > 1e-30


def test_saturated_noisy_edfa_rejects_nonphysical_parameters_and_real_fields() -> None:
    with pytest.raises(ValueError, match="saturation_power"):
        SaturatedNoisyEDFA(SimulationGrid(128.0), saturation_power=0.0)
    with pytest.raises(ValueError, match="noise_figure_db"):
        SaturatedNoisyEDFA(SimulationGrid(128.0), noise_figure_db=-1.0)
    amplifier = SaturatedNoisyEDFA(SimulationGrid(128.0), gain_bandwidth_hz=128.0)
    with pytest.raises(TypeError, match="complex"):
        amplifier(torch.ones(128))
    with torch.no_grad():
        amplifier.saturation_power.fill_(-1.0)
    with pytest.raises(ValueError, match="saturation_power"):
        amplifier(torch.ones(128, dtype=torch.complex64))
