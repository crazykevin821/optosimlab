import torch

from optosimlab.devices.mzi import FourInputMZI


def test_four_input_mzi_has_unitary_port_normalization() -> None:
    mzi = FourInputMZI(trainable=False)
    one_active = torch.zeros(4, 32, dtype=torch.complex64)
    one_active[0] = 1.0
    all_active = torch.ones(4, 32, dtype=torch.complex64)
    assert torch.allclose(mzi(one_active).abs().square().mean(), torch.tensor(0.25))
    assert torch.allclose(mzi(all_active).abs().square().mean(), torch.tensor(4.0))


def test_four_input_mzi_phase_can_create_destructive_interference() -> None:
    mzi = FourInputMZI(torch.tensor([0.0, torch.pi, 0.0, torch.pi]), trainable=False)
    assert torch.max(mzi(torch.ones(4, 32, dtype=torch.complex64)).abs()) < 2e-7
