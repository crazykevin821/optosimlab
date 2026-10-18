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
