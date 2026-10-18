import copy

import torch

from optosimlab import DifferentialPhotonicConvolution, PhotonicRealConvolution


def test_real_photonic_convolution_chunking_matches_full_batch_tensor() -> None:
    torch.manual_seed(17)
    layer = PhotonicRealConvolution(
        torch.tensor([0.2, 0.5, 0.8]),
        sample_rate_hz=100e9,
        line_spacing_hz=10e9,
        tap_delay_samples=2,
        trainable=False,
    )
    samples = torch.rand(3, 2, 47)
    expected = layer(samples)
    actual = layer.forward_chunked(samples, chunk_size=11)
    assert actual.shape == samples.shape
    assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-6)


def test_chunked_path_keeps_input_and_waveshaper_gradients() -> None:
    torch.manual_seed(18)
    full_layer = PhotonicRealConvolution(torch.tensor([0.25, 0.6]), trainable=True)
    chunked_layer = copy.deepcopy(full_layer)
    full_input = torch.rand(2, 31, requires_grad=True)
    chunked_input = full_input.detach().clone().requires_grad_(True)
    full_layer(full_input).sum().backward()
    chunked_layer.forward_chunked(chunked_input, 7).sum().backward()
    assert torch.allclose(chunked_input.grad, full_input.grad, atol=3e-6, rtol=3e-6)
    assert torch.allclose(
        chunked_layer.waveshaper.attenuation_db.grad,
        full_layer.waveshaper.attenuation_db.grad,
        atol=3e-6,
        rtol=3e-6,
    )


def test_differential_branches_clone_supplied_physical_devices() -> None:
    layer = DifferentialPhotonicConvolution(torch.tensor([-0.2, 0.4]), trainable=False)
    assert layer.positive.modulator is not layer.negative.modulator
    assert layer.positive.fiber is not layer.negative.fiber
    assert layer.positive.detector is not layer.negative.detector


def test_projection_keeps_trainable_waveshaper_passive() -> None:
    layer = DifferentialPhotonicConvolution(torch.tensor([-0.2, 0.4]))
    with torch.no_grad():
        layer.positive.waveshaper.attenuation_db.fill_(-1.0)
        layer.negative.waveshaper.attenuation_db.fill_(-2.0)
    layer.project_physical_parameters_()
    assert torch.all(layer.positive.waveshaper.attenuation_db >= 0)
    assert torch.all(layer.negative.waveshaper.attenuation_db >= 0)
