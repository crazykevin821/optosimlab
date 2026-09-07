import torch

from optosimlab import DifferentialDualPortMZM, FourPortMZM, IdealMZM


def test_ideal_mzm_peak_and_null_follow_shared_interferometer() -> None:
    field = torch.ones(32, dtype=torch.complex64)
    mzm = IdealMZM(v_pi=2.0, trainable=False)
    assert torch.allclose(mzm(field, torch.tensor(0.0)), field, atol=2e-6)
    assert mzm(field, torch.tensor(2.0)).abs().max() < 2e-6
    assert isinstance(mzm.interferometer, DifferentialDualPortMZM)


def test_four_port_mzm_is_lossless_for_arbitrary_two_port_input() -> None:
    torch.manual_seed(17)
    input0 = torch.complex(torch.randn(64), torch.randn(64))
    input1 = torch.complex(torch.randn(64), torch.randn(64))
    voltage = torch.linspace(-1.0, 1.0, 64)
    mzm = FourPortMZM(v_pi=2.0, trainable=False)
    output0, output1 = mzm(input0, input1, voltage)
    input_energy = input0.abs().square() + input1.abs().square()
    output_energy = output0.abs().square() + output1.abs().square()
    assert torch.allclose(output_energy, input_energy, rtol=2e-6, atol=2e-6)
