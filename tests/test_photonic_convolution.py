import math

import torch
import pytest

from optosimlab import (
    DifferentialPhotonicConvolution,
    IdealMZM,
    LinearDispersiveFiber,
    PracticalMZM,
    PhotodetectorSum,
    PhotonicRealConvolution,
    SimulationGrid,
    WaveShaper,
)


def causal_convolution(samples: torch.Tensor, weights: torch.Tensor, delay: int = 1) -> torch.Tensor:
    result = torch.zeros_like(samples)
    for tap, weight in enumerate(weights):
        shift = tap * delay
        if shift < samples.shape[-1]:
            result[..., shift:] += weight * samples[..., : samples.shape[-1] - shift]
    return result


def test_nature_route_matches_pointwise_real_convolution_and_physical_power() -> None:
    weights = torch.tensor([0.2, 0.5, 0.8])
    samples = torch.tensor([0.1, 0.3, 0.7, 1.0, 0.2, 0.6, 0.4, 0.9])
    layer = PhotonicRealConvolution(weights, sample_rate_hz=100e9, line_spacing_hz=10e9, line_power_w=2e-3)
    expected = causal_convolution(samples, weights)
    assert layer.optical_chain._is_graph
    assert isinstance(layer.waveshaper, WaveShaper)
    assert isinstance(layer.modulator, IdealMZM)
    assert isinstance(layer.fiber, LinearDispersiveFiber)
    assert isinstance(layer.detector, PhotodetectorSum)
    assert tuple(layer.optical_chain.graph_devices) == ("waveshaper", "modulator", "fiber", "photodetector")
    assert torch.allclose(layer(samples), expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(layer.detected_power_w(samples), 2e-3 * expected, atol=2e-9, rtol=2e-6)


def test_photonic_convolution_supports_batches_delay_and_gradients() -> None:
    weights = torch.tensor([0.25, 0.5])
    samples = torch.tensor([[0.2, 0.4, 0.8, 0.6, 0.3], [0.9, 0.7, 0.5, 0.3, 0.1]], requires_grad=True)
    layer = PhotonicRealConvolution(weights, sample_rate_hz=100e9, line_spacing_hz=10e9, tap_delay_samples=2)
    output = layer(samples)
    assert torch.allclose(output, causal_convolution(samples, weights, 2), atol=2e-6)
    output.sum().backward()
    assert samples.grad is not None and torch.all(torch.isfinite(samples.grad))
    assert layer.waveshaper.attenuation_db.grad is not None


def test_differential_branches_match_signed_real_convolution() -> None:
    weights = torch.tensor([-0.4, 0.2, -0.8])
    samples = torch.tensor([0.2, 0.5, 0.7, 0.3, 0.9])
    layer = DifferentialPhotonicConvolution(weights, sample_rate_hz=100e9, line_spacing_hz=10e9, trainable=False)
    assert torch.allclose(layer(samples), causal_convolution(samples, weights), atol=3e-6)


def test_practical_mzm_path_calibrates_loss_and_extinction_floor() -> None:
    weights = torch.tensor([0.25, 0.75])
    samples = torch.tensor([0.0, 0.3, 1.0, 0.2])
    modulator = PracticalMZM(
        v_pi=2.0,
        insertion_loss_db=3.0,
        extinction_ratio_db=20.0,
        trainable=False,
    )
    layer = PhotonicRealConvolution(weights, modulator=modulator, trainable=False)
    expected = causal_convolution(samples, weights)
    assert torch.allclose(layer(samples), expected, atol=3e-6, rtol=3e-6)

    minimum, maximum = modulator.power_transmission_bounds(samples)
    physical_expected = layer.line_power_w * (
        minimum * weights.sum() + (maximum - minimum) * expected
    )
    assert torch.allclose(layer.detected_power_w(samples), physical_expected, atol=2e-9, rtol=3e-6)


def test_lossy_fiber_is_in_physical_power_but_calibrates_out_of_convolution() -> None:
    sample_rate_hz = 100e9
    line_spacing_hz = 10e9
    beta2 = LinearDispersiveFiber.beta2_from_dispersion(17.0, 1550e-9)
    delay_s = 1.0 / sample_rate_hz
    length_m = delay_s / (-beta2 * 2.0 * math.pi * line_spacing_hz)
    fiber = LinearDispersiveFiber(
        SimulationGrid(sample_rate_hz),
        length_m=length_m,
        attenuation_db_per_km=0.2,
        beta2_s2_per_m=beta2,
        trainable=False,
    )
    weights = torch.tensor([0.25, 0.75])
    samples = torch.tensor([0.0, 0.3, 1.0, 0.2])
    modulator = PracticalMZM(
        v_pi=2.0,
        insertion_loss_db=3.0,
        extinction_ratio_db=20.0,
        trainable=False,
    )
    layer = PhotonicRealConvolution(
        weights,
        sample_rate_hz=sample_rate_hz,
        line_spacing_hz=line_spacing_hz,
        modulator=modulator,
        fiber=fiber,
        trainable=False,
    )
    expected = causal_convolution(samples, weights)
    assert torch.allclose(layer(samples), expected, atol=3e-6, rtol=3e-6)

    minimum, maximum = modulator.power_transmission_bounds(samples)
    fiber_transmission = fiber.power_transmission(samples)
    physical_expected = layer.line_power_w * fiber_transmission * (
        minimum * weights.sum() + (maximum - minimum) * expected
    )
    assert torch.allclose(layer.detected_power_w(samples), physical_expected, atol=2e-9, rtol=3e-6)


def test_single_branch_rejects_negative_or_integer_samples() -> None:
    layer = PhotonicRealConvolution(torch.tensor([0.5]), trainable=False)
    with pytest.raises(ValueError, match="nonnegative"):
        layer(torch.tensor([0.2, -0.1]))
    with pytest.raises(TypeError, match="floating-point"):
        layer(torch.tensor([1, 2]))
