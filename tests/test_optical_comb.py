import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.comb import OpticalFrequencyComb


def test_comb_places_power_and_phase_at_expected_fft_bins() -> None:
    comb = OpticalFrequencyComb(
        SimulationGrid(sample_rate_hz=64.0),
        line_spacing_hz=8.0,
        line_count=3,
        line_powers=torch.tensor([1.0, 4.0, 9.0]),
        line_phases_rad=torch.tensor([0.0, torch.pi / 2, torch.pi]),
        trainable=False,
    )
    spectrum = torch.fft.fft(comb(64))
    expected = 64.0 * torch.sqrt(torch.tensor([1.0, 4.0, 9.0])) * torch.exp(1j * torch.tensor([0.0, torch.pi / 2, torch.pi]))
    assert torch.allclose(spectrum[torch.tensor([56, 0, 8])], expected, atol=2e-5, rtol=2e-6)
    assert torch.count_nonzero(spectrum.abs() > 1e-5) == 3


def test_comb_mean_power_is_sum_of_distinct_line_powers() -> None:
    comb = OpticalFrequencyComb(
        SimulationGrid(sample_rate_hz=128.0),
        line_spacing_hz=16.0,
        line_count=5,
        line_powers=torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0]),
        trainable=False,
    )
    assert torch.allclose(comb(128).abs().square().mean(), torch.tensor(10.5), rtol=2e-6, atol=2e-6)
    assert torch.allclose(comb.total_power, torch.tensor(10.5))


def test_comb_line_power_and_phase_are_trainable() -> None:
    comb = OpticalFrequencyComb(
        SimulationGrid(sample_rate_hz=64.0),
        line_spacing_hz=8.0,
        line_count=1,
        line_powers=2.0,
        line_phases_rad=torch.tensor([0.2]),
    )
    loss = comb(64).real.mean()
    loss.backward()
    assert comb.line_powers.grad is not None
    assert comb.line_phases_rad.grad is not None
    assert comb.line_powers.grad.abs().item() > 1e-6
    assert comb.line_phases_rad.grad.abs().item() > 1e-6


def test_comb_rejects_off_bin_spacing() -> None:
    comb = OpticalFrequencyComb(SimulationGrid(sample_rate_hz=64.0), line_spacing_hz=7.5)
    with pytest.raises(ValueError, match="integer multiple"):
        comb(64)


def test_comb_rejects_lines_outside_representable_bandwidth() -> None:
    comb = OpticalFrequencyComb(SimulationGrid(sample_rate_hz=64.0), line_spacing_hz=10.0, line_count=9)
    with pytest.raises(ValueError, match="exceed"):
        comb(64)
