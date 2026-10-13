import math

import torch

from optosimlab.config import SimulationGrid
from optosimlab.devices.fiber import LinearDispersiveFiber


def test_dispersion_conversion_round_trips() -> None:
    wavelength = 1550e-9
    beta2 = LinearDispersiveFiber.beta2_from_dispersion(17.0, wavelength)
    assert math.isclose(LinearDispersiveFiber.dispersion_from_beta2(beta2, wavelength), 17.0, rel_tol=1e-12)
    assert beta2 < 0.0  # standard positive D at 1550 nm implies anomalous beta2


def test_lossless_linear_fiber_conserves_discrete_energy() -> None:
    torch.manual_seed(7)
    field = torch.complex(torch.randn(256), torch.randn(256))
    fiber = LinearDispersiveFiber(SimulationGrid(100e9), length_m=20_000.0, beta2_s2_per_m=-21e-27)
    output = fiber(field)
    assert torch.allclose(output.abs().square().sum(), field.abs().square().sum(), rtol=2e-6, atol=2e-5)


def test_fiber_power_loss_uses_db_per_km_correctly() -> None:
    field = torch.ones(64, dtype=torch.complex64)
    fiber = LinearDispersiveFiber(SimulationGrid(100e9), length_m=1_000.0, attenuation_db_per_km=10.0)
    output_power = fiber(field).abs().square().mean()
    assert torch.allclose(output_power, torch.tensor(0.1), rtol=1e-5, atol=1e-6)
    assert torch.allclose(fiber.power_transmission(field.real), torch.tensor(0.1), rtol=1e-6)


def test_fiber_phase_matches_beta2_transfer_formula() -> None:
    grid = SimulationGrid(64.0)
    beta2 = -0.02
    length = 3.0
    fiber = LinearDispersiveFiber(grid, length_m=length, beta2_s2_per_m=beta2)
    response = fiber.transfer_function(64, device=torch.device("cpu"), dtype=torch.float64)
    omega = 2.0 * math.pi * 8.0
    expected = complex(math.cos(0.5 * beta2 * omega**2 * length), math.sin(0.5 * beta2 * omega**2 * length))
    assert torch.allclose(response[8], torch.tensor(expected, dtype=torch.complex128), atol=1e-12)


def test_wdm_dispersion_gives_progressive_positive_delay_without_mixing_channels() -> None:
    sample_rate = 8.0
    spacing = 1.0
    beta2 = -1.0
    delay_s = 1.0 / sample_rate
    length = delay_s / (-beta2 * 2.0 * math.pi * spacing)
    fiber = LinearDispersiveFiber(SimulationGrid(sample_rate), length_m=length, beta2_s2_per_m=beta2)
    channels = torch.zeros(2, 8, dtype=torch.complex64)
    channels[:, 0] = 1.0
    output = fiber(channels, torch.tensor([0.0, spacing]))
    assert output[0].abs().argmax().item() == 0
    assert output[1].abs().argmax().item() == 1
    assert torch.allclose(output.abs().square().sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(fiber.adjacent_channel_delay_s(spacing), torch.tensor(delay_s), atol=1e-7)
