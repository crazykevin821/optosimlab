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


def test_differential_encoding_preserves_kernel_and_avoids_dead_branch_gradients() -> None:
    weights = torch.tensor([-0.3, 0.0, 0.3])
    layer = DifferentialPhotonicConvolution(weights)
    assert torch.allclose(layer.effective_weights, weights, atol=2e-6, rtol=2e-6)
    samples = torch.rand(8, 31)
    layer(samples).square().mean().backward()
    positive_grad = layer.positive.waveshaper.attenuation_db.grad
    negative_grad = layer.negative.waveshaper.attenuation_db.grad
    assert positive_grad is not None and torch.all(positive_grad.abs() > 1e-8)
    assert negative_grad is not None and torch.all(negative_grad.abs() > 1e-8)


def test_differential_weight_can_cross_zero_during_optimisation() -> None:
    layer = DifferentialPhotonicConvolution(torch.tensor([0.2]))
    optimiser = torch.optim.Adam(layer.parameters(), lr=0.1)
    target = torch.tensor([-0.2])
    for _ in range(160):
        optimiser.zero_grad()
        (layer.effective_weights - target).square().sum().backward()
        optimiser.step()
        layer.project_physical_parameters_()
    assert layer.effective_weights.item() < -0.1


def test_all_zero_differential_kernel_keeps_trainable_scale() -> None:
    layer = DifferentialPhotonicConvolution(torch.zeros(3))
    assert torch.allclose(layer.effective_weights, torch.zeros(3), atol=2e-6)
    target = torch.tensor([0.2, -0.1, 0.3])
    (layer.effective_weights - target).square().sum().backward()
    positive_grad = layer.positive.waveshaper.attenuation_db.grad
    negative_grad = layer.negative.waveshaper.attenuation_db.grad
    assert positive_grad is not None and torch.all(positive_grad.abs() > 1e-5)
    assert negative_grad is not None and torch.all(negative_grad.abs() > 1e-5)


def test_iter_chunked_can_be_consumed_one_chunk_at_a_time() -> None:
    layer = DifferentialPhotonicConvolution(torch.tensor([-0.2, 0.4]))
    samples = torch.rand(2, 37, requires_grad=True)
    detached_outputs = []
    for chunk in layer.iter_chunked(samples, 9):
        detached_outputs.append(chunk.detach())
        chunk.square().mean().backward()
    assert torch.allclose(torch.cat(detached_outputs, dim=-1), layer(samples), atol=3e-6, rtol=3e-6)
    assert samples.grad is not None and torch.all(torch.isfinite(samples.grad))
