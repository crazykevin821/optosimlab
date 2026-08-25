import torch

from optosimlab.devices.noise import AdditiveComplexGaussianNoise, OpticalAttenuator


def test_attenuator_uses_power_db_definition() -> None:
    field = torch.ones(64, dtype=torch.complex64)
    output = OpticalAttenuator(attenuation_db=6.0)(field)
    assert torch.allclose(output.abs().square().mean(), torch.tensor(10 ** (-0.6)), rtol=1e-5)


def test_zero_noise_preserves_field_exactly() -> None:
    torch.manual_seed(1)
    field = torch.complex(torch.randn(32), torch.randn(32))
    assert torch.equal(AdditiveComplexGaussianNoise(0.0)(field), field)


def test_noise_power_matches_sigma_squared_statistically() -> None:
    torch.manual_seed(2)
    samples = 200_000
    field = torch.zeros(samples, dtype=torch.complex64)
    sigma = 0.25
    noise = AdditiveComplexGaussianNoise(sigma)(field)
    assert torch.allclose(noise.abs().square().mean(), torch.tensor(sigma**2), rtol=0.02)
