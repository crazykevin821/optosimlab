import torch

from optosimlab.devices.noise import AdditiveComplexGaussianNoise, OpticalAttenuator
from optosimlab.devices.photodetector import NoisyPhotodetectorSum


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


def test_noisy_photodetector_zero_noise_is_exact_and_nonzero_noise_has_expected_scale() -> None:
    channels = torch.ones(2, 50_000, dtype=torch.complex64)
    exact = NoisyPhotodetectorSum(noise_std_w=0.0)(channels)
    assert torch.equal(exact, torch.full((50_000,), 2.0))
    torch.manual_seed(23)
    noisy = NoisyPhotodetectorSum(noise_std_w=0.02)(channels)
    error = noisy - 2.0
    assert abs(float(error.mean())) < 5e-4
    assert abs(float(error.std()) - 0.02) < 5e-4
