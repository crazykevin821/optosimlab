import torch
from torch.nn import functional as F

from optosimlab.blocks.complex_conv import ComplexConv1d


def test_complex_conv_matches_complex_multiplication_and_adds_bias_once() -> None:
    layer = ComplexConv1d(1, 1, 1, bias=True)
    with torch.no_grad():
        layer.real.weight.fill_(2.0)
        layer.imag.weight.fill_(3.0)
        layer.real.bias.fill_(5.0)
        layer.imag.bias.fill_(7.0)
    field = torch.complex(torch.tensor([[[1.0, -2.0]]]), torch.tensor([[[4.0, 6.0]]]))
    output = layer(field)
    expected_real = F.conv1d(field.real, layer.real.weight, layer.real.bias) - F.conv1d(field.imag, layer.imag.weight)
    expected_imag = F.conv1d(field.imag, layer.real.weight, layer.imag.bias) + F.conv1d(field.real, layer.imag.weight)
    assert torch.allclose(output, torch.complex(expected_real, expected_imag))
