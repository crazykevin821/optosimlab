import pytest
import torch

from optosimlab.config import SimulationGrid


def test_frequency_grid_matches_torch_fft_convention() -> None:
    grid = SimulationGrid(sample_rate_hz=8.0)
    frequency = grid.frequencies_hz(8)
    assert torch.equal(frequency, torch.tensor([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]))
    assert grid.dt_s == 0.125


@pytest.mark.parametrize("sample_rate", [0.0, -1.0])
def test_grid_rejects_non_positive_sample_rate(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        SimulationGrid(sample_rate_hz=sample_rate)


@pytest.mark.parametrize("samples", [0, -2])
def test_grid_rejects_non_positive_sample_count(samples: int) -> None:
    grid = SimulationGrid(sample_rate_hz=1.0)
    with pytest.raises(ValueError, match="samples"):
        grid.frequencies_hz(samples)
