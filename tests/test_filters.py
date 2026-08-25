import math

import pytest
import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.filters import FrequencyDomainLowPass


def complex_tone(sample_rate_hz: float, samples: int, tone_hz: float) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / sample_rate_hz
    phase = 2.0 * torch.pi * tone_hz * time
    return torch.complex(torch.cos(phase), torch.sin(phase))


def test_low_pass_matches_analytic_gain_at_fft_bin() -> None:
    sample_rate = 64.0
    tone_hz = 8.0
    cutoff_hz = 4.0
    field = complex_tone(sample_rate, 64, tone_hz)
    low_pass = FrequencyDomainLowPass(SimulationGrid(sample_rate), cutoff_hz, order=2)
    output = low_pass(field)
    expected_gain = 1.0 / math.sqrt(1.0 + (tone_hz / cutoff_hz) ** 4)
    assert torch.allclose(output, field * expected_gain, atol=2e-6, rtol=2e-6)


def test_cutoff_is_minus_three_db_power_point() -> None:
    low_pass = FrequencyDomainLowPass(SimulationGrid(64.0), cutoff_hz=8.0)
    gain = low_pass.transfer_function(64, device=torch.device("cpu"), dtype=torch.float32)[8]
    assert torch.allclose(gain.square(), torch.tensor(0.5), atol=1e-6)


def test_low_pass_rejects_real_field() -> None:
    low_pass = FrequencyDomainLowPass(SimulationGrid(64.0), cutoff_hz=8.0)
    with pytest.raises(TypeError, match="complex"):
        low_pass(torch.ones(64))
