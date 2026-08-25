import math

import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.edfa import SmallSignalEDFA


def test_edfa_peak_gain_has_correct_current_mapping_and_power_db_loss() -> None:
    amplifier = SmallSignalEDFA(
        SimulationGrid(128.0),
        pump_current_ma=100.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=32.0,
        insertion_loss_db=3.0,
        trainable=False,
    )
    expected_gain_db = 0.5 * (100.0 - 20.0) - 3.0
    assert torch.allclose(amplifier.peak_net_gain_db(), torch.tensor(expected_gain_db))
    output_power = amplifier(torch.ones(128, dtype=torch.complex64)).abs().square().mean()
    assert torch.allclose(output_power, torch.tensor(10 ** (expected_gain_db / 10.0)), rtol=2e-5)


def test_edfa_gain_bandwidth_is_power_gain_fwhm() -> None:
    amplifier = SmallSignalEDFA(
        SimulationGrid(128.0),
        pump_current_ma=20.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=1.0,
        gain_bandwidth_hz=32.0,
        trainable=False,
    )
    gain = amplifier.power_gain(128)
    assert torch.allclose(gain[16] / gain[0], torch.tensor(0.5), rtol=2e-6, atol=2e-6)
    assert torch.allclose(gain[-16] / gain[0], torch.tensor(0.5), rtol=2e-6, atol=2e-6)


def test_edfa_applies_gain_at_configured_frequency_offset_without_phase_change() -> None:
    grid = SimulationGrid(128.0)
    amplifier = SmallSignalEDFA(
        grid,
        pump_current_ma=60.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=0.5,
        gain_bandwidth_hz=32.0,
        gain_center_offset_hz=8.0,
        trainable=False,
    )
    samples = 128
    tone = torch.exp(2j * torch.pi * 8.0 * torch.arange(samples) / grid.sample_rate_hz)
    output = amplifier(tone)
    expected_power_gain = 10 ** (20.0 / 10.0)
    assert torch.allclose(output.abs().square().mean(), torch.tensor(expected_power_gain), rtol=2e-5)
    assert torch.max((output / tone).imag.abs()) < 2e-5


def test_edfa_parameters_are_registered_and_differentiable() -> None:
    grid = SimulationGrid(128.0)
    amplifier = SmallSignalEDFA(grid, pump_current_ma=60.0, transparency_current_ma=20.0, gain_slope_db_per_ma=0.5, gain_bandwidth_hz=32.0)
    field = torch.exp(2j * torch.pi * 8.0 * torch.arange(128) / grid.sample_rate_hz)
    amplifier(field).abs().mean().backward()
    assert set(dict(amplifier.named_parameters())) == {
        "pump_current_ma",
        "transparency_current_ma",
        "gain_slope_db_per_ma",
        "gain_bandwidth_hz",
        "gain_center_offset_hz",
        "insertion_loss_db",
    }
    for parameter in amplifier.parameters():
        assert parameter.grad is not None
        assert parameter.grad.abs().item() > 1e-8


def test_edfa_rejects_invalid_physical_values_and_real_fields() -> None:
    with pytest.raises(ValueError, match="gain_bandwidth_hz must be positive"):
        SmallSignalEDFA(SimulationGrid(128.0), gain_bandwidth_hz=0.0)
    with pytest.raises(ValueError, match="pump_current_ma"):
        SmallSignalEDFA(SimulationGrid(128.0), pump_current_ma=-1.0)
    amplifier = SmallSignalEDFA(SimulationGrid(128.0), gain_bandwidth_hz=32.0)
    with pytest.raises(TypeError, match="complex"):
        amplifier(torch.ones(128))
    with torch.no_grad():
        amplifier.gain_bandwidth_hz.fill_(-1.0)
    with pytest.raises(ValueError, match="gain_bandwidth_hz"):
        amplifier(torch.ones(128, dtype=torch.complex64))


def test_edfa_below_transparency_has_no_current_induced_gain() -> None:
    amplifier = SmallSignalEDFA(
        SimulationGrid(128.0),
        pump_current_ma=10.0,
        transparency_current_ma=20.0,
        gain_slope_db_per_ma=1.0,
        gain_bandwidth_hz=32.0,
        trainable=False,
    )
    assert math.isclose(amplifier.peak_net_gain_db().item(), 0.0, abs_tol=1e-7)
