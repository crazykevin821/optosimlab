import torch

from optosimlab import MZIComplexConv1d


def direct_causal_complex_convolution(field: torch.Tensor, weights: torch.Tensor, delay: int) -> torch.Tensor:
    output = torch.zeros(
        field.shape[0], weights.shape[0], field.shape[-1], dtype=field.dtype, device=field.device
    )
    for tap in range(weights.shape[-1]):
        shift = tap * delay
        if shift < field.shape[-1]:
            output[..., shift:] += torch.einsum(
                "ncl,oc->nol", field[..., : field.shape[-1] - shift], weights[..., tap]
            )
    return output


def test_mzi_complex_convolution_matches_direct_complex_mac() -> None:
    torch.manual_seed(19)
    weights = torch.complex(torch.randn(2, 3, 4), torch.randn(2, 3, 4)) * 0.2
    field = torch.complex(torch.randn(5, 3, 29), torch.randn(5, 3, 29))
    layer = MZIComplexConv1d(weights, tap_delay_samples=2, trainable=False)
    assert torch.allclose(layer.effective_weights, weights, atol=2e-6, rtol=2e-6)
    expected = direct_causal_complex_convolution(field, weights, 2)
    assert torch.allclose(layer(field), expected, atol=3e-6, rtol=3e-6)
    assert torch.allclose(layer.forward_chunked(field, 8), expected, atol=3e-6, rtol=3e-6)


def test_mzi_complex_convolution_supports_unbatched_and_autograd() -> None:
    weights = torch.tensor([[[0.2 + 0.1j, -0.3 + 0.4j]]], dtype=torch.complex64)
    layer = MZIComplexConv1d(weights, bias=True)
    field = torch.ones(1, 13, dtype=torch.complex64, requires_grad=True)
    output = layer(field)
    assert output.shape == (1, 13)
    output.abs().square().mean().backward()
    assert field.grad is not None
    assert layer.weight_bank.theta_rad.grad is not None
    assert layer.weight_bank.phi_rad.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None


def test_mzi_complex_convolution_rejects_real_input() -> None:
    layer = MZIComplexConv1d(torch.ones(1, 1, 2, dtype=torch.complex64))
    try:
        layer(torch.ones(1, 1, 8))
    except TypeError as error:
        assert "complex" in str(error)
    else:
        raise AssertionError("real optical fields must be rejected")


def test_mzi_complex_projection_enforces_passive_row_norm() -> None:
    layer = MZIComplexConv1d(torch.ones(2, 2, 3, dtype=torch.complex64) * (0.1 + 0.2j))
    with torch.no_grad():
        layer.weight_bank.theta_rad.zero_()
        layer.weight_bank.insertion_loss_db.fill_(-1.0)
    layer.project_physical_parameters_()
    normalized = layer.effective_weights / layer.weight_bank.weight_scale
    row_norms = torch.linalg.vector_norm(normalized, dim=(1, 2))
    assert torch.all(row_norms <= 1.0 + 2e-6)
    assert layer.weight_bank.insertion_loss_db >= 0


def test_chunked_complex_convolution_reuses_one_phase_error_sample() -> None:
    weights = torch.tensor([[[0.3 + 0.2j, -0.1 + 0.4j]]])
    layer = MZIComplexConv1d(weights, phase_error_std_rad=0.1)
    field = torch.randn(2, 1, 31, dtype=torch.complex64)
    torch.manual_seed(91)
    expected = layer(field)
    torch.manual_seed(91)
    actual = layer.forward_chunked(field, 7)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
