import math

import pytest
import torch

from optosimlab import (
    SimulationGrid,
    centered_frequency_axis_hz,
    error_vector_magnitude,
    eye_diagram,
    gaussian_ber_estimate,
    gaussian_q_factor,
    mean_power,
    power_spectrum,
)


def test_power_spectrum_obeys_parseval_and_frequency_axis_alignment() -> None:
    samples = 16
    field = torch.exp(2j * torch.pi * 4 * torch.arange(samples) / samples)
    spectrum = power_spectrum(field)
    frequencies = centered_frequency_axis_hz(SimulationGrid(16.0), samples)
    assert torch.allclose(spectrum.sum(), mean_power(field), atol=2e-6)
    assert frequencies[torch.argmax(spectrum)] == 4.0


def test_eye_diagram_returns_overlapping_two_symbol_traces() -> None:
    traces = eye_diagram(torch.arange(12), samples_per_symbol=2)
    assert traces.shape == (5, 4)
    assert torch.equal(traces[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(traces[-1], torch.tensor([8, 9, 10, 11]))


def test_evm_is_rms_normalized_complex_error() -> None:
    reference = torch.ones(32, dtype=torch.complex64)
    estimate = reference * (1.0 + 0.1j)
    assert torch.allclose(error_vector_magnitude(reference, estimate), torch.tensor(0.1), atol=2e-6)


def test_gaussian_q_and_ber_match_analytic_definition() -> None:
    level_zero = torch.tensor([0.0, 0.0, 1.0, 1.0])
    level_one = torch.tensor([3.0, 3.0, 4.0, 4.0])
    q = gaussian_q_factor(level_zero, level_one)
    assert torch.allclose(q, torch.tensor(3.0))
    assert torch.allclose(gaussian_ber_estimate(level_zero, level_one), torch.tensor(0.5 * math.erfc(3.0 / math.sqrt(2.0))), atol=2e-8)


def test_system_metrics_reject_invalid_shapes_and_nonreal_decision_levels() -> None:
    with pytest.raises(ValueError, match="same shape"):
        error_vector_magnitude(torch.ones(4, dtype=torch.complex64), torch.ones(5, dtype=torch.complex64))
    with pytest.raises(ValueError, match="two-symbol"):
        eye_diagram(torch.ones(3), samples_per_symbol=2)
    with pytest.raises(TypeError, match="real"):
        gaussian_ber_estimate(torch.ones(2, dtype=torch.complex64), torch.ones(2))
