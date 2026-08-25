import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.waveshaper import WaveShaper


def test_flat_waveshaper_is_identity() -> None:
    torch.manual_seed(5)
    field = torch.complex(torch.randn(64), torch.randn(64))
    shaper = WaveShaper(SimulationGrid(64.0), control_points=64, trainable=False)
    assert torch.allclose(shaper(field), field, rtol=2e-6, atol=2e-6)


def test_uniform_waveshaper_attenuation_has_correct_power_loss() -> None:
    shaper = WaveShaper(SimulationGrid(64.0), control_points=64, trainable=False)
    with torch.no_grad():
        shaper.attenuation_db.fill_(20.0)
    output = shaper(torch.ones(64, dtype=torch.complex64))
    assert torch.allclose(output.abs().square().mean(), torch.tensor(0.01), rtol=1e-5)


def test_uniform_phase_is_unit_magnitude_phase_rotation() -> None:
    shaper = WaveShaper(SimulationGrid(64.0), control_points=64, trainable=False)
    with torch.no_grad():
        shaper.phase_rad.fill_(torch.pi / 2)
    output = shaper(torch.ones(64, dtype=torch.complex64))
    expected = torch.full((64,), 1j, dtype=torch.complex64)
    assert torch.allclose(output, expected, atol=2e-6)
