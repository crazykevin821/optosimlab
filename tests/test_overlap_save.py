import pytest
import torch

from optosimlab.blocks.overlap_save import OverlapSaveFIR


def direct_causal_fir(field: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(field)
    for tap, coefficient in enumerate(impulse):
        output[..., tap:] += coefficient * field[..., : field.shape[-1] - tap]
    return output


def test_overlap_save_matches_direct_complex_linear_convolution() -> None:
    field = torch.tensor([1 + 2j, -1j, 2 - 1j, 3 + 0j, -2j, 1 + 1j], dtype=torch.complex64)
    impulse = torch.tensor([1 + 0j, 0.5 - 0.25j, -0.1j], dtype=torch.complex64)
    module = OverlapSaveFIR(impulse, fft_size=4)
    assert torch.allclose(module(field), direct_causal_fir(field, impulse), atol=2e-6)


def test_overlap_save_matches_direct_convolution_for_batch_and_partial_final_block() -> None:
    torch.manual_seed(4)
    field = torch.randn(2, 3, 23, dtype=torch.complex64)
    impulse = torch.randn(5, dtype=torch.complex64)
    module = OverlapSaveFIR(impulse, fft_size=8)
    assert torch.allclose(module(field), direct_causal_fir(field, impulse), atol=3e-6, rtol=3e-6)


def test_overlap_save_has_trainable_complex_impulse_response() -> None:
    impulse = torch.tensor([1 + 0j, 0.2 + 0.1j, -0.1j], dtype=torch.complex64)
    module = OverlapSaveFIR(impulse, fft_size=8, trainable=True)
    field = torch.randn(17, dtype=torch.complex64, requires_grad=True)
    module(field).abs().square().mean().backward()
    assert module.impulse_response.grad is not None
    assert module.impulse_response.grad.abs().sum() > 1e-7
    assert field.grad is not None


def test_overlap_save_frequency_response_and_block_geometry_are_correct() -> None:
    impulse = torch.tensor([1 + 0j, 1 + 0j], dtype=torch.complex64)
    module = OverlapSaveFIR(impulse, fft_size=8)
    assert module.valid_samples_per_block == 7
    assert torch.allclose(module.frequency_response()[0], torch.tensor(2 + 0j))


def test_overlap_save_rejects_invalid_filter_contracts() -> None:
    with pytest.raises(TypeError, match="complex"):
        OverlapSaveFIR(torch.ones(2), fft_size=4)
    with pytest.raises(ValueError, match="greater"):
        OverlapSaveFIR(torch.ones(4, dtype=torch.complex64), fft_size=3)
    module = OverlapSaveFIR(torch.ones(2, dtype=torch.complex64), fft_size=4)
    with pytest.raises(TypeError, match="complex"):
        module(torch.ones(8))
