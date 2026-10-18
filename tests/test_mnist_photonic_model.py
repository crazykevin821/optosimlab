import torch
import pytest
from torch import nn

from optosimlab import IntensityMZMActivation, PhotonicMNISTClassifier


def test_ideal_mnist_model_accepts_image_batches_and_backpropagates() -> None:
    torch.manual_seed(20)
    model = PhotonicMNISTClassifier(image_size=4, kernel_size=3, mode="ideal", activation="relu")
    images = torch.rand(6, 1, 4, 4)
    labels = torch.randint(0, 10, (6,))
    logits = model(images)
    assert logits.shape == (6, 10)
    nn.CrossEntropyLoss()(logits, labels).backward()
    assert model.classifier.weight.grad is not None
    assert model.optical_convolution.positive.waveshaper.attenuation_db.grad is not None
    assert model.optical_convolution.negative.waveshaper.attenuation_db.grad is not None


def test_mnist_model_chunking_and_mzi_activation() -> None:
    torch.manual_seed(21)
    model = PhotonicMNISTClassifier(image_size=4, kernel_size=3, activation="mzi")
    assert isinstance(model.activation, IntensityMZMActivation)
    images = torch.rand(3, 1, 4, 4)
    full = model(images)
    model.chunk_size = 5
    chunked = model(images)
    assert torch.allclose(chunked, full, atol=5e-6, rtol=5e-6)


def test_nonideal_mnist_model_declares_and_executes_nonidealities() -> None:
    torch.manual_seed(22)
    model = PhotonicMNISTClassifier(
        image_size=4,
        kernel_size=3,
        mode="nonideal",
        detector_noise_std_w=1e-7,
    )
    description = model.nonideality_description
    assert description["mzm_insertion_loss_db"] == 3.0
    assert description["mzm_extinction_ratio_db"] == 20.0
    assert description["fiber_loss_db_per_km"] == 0.2
    assert description["detector_noise_std_w"] == pytest.approx(1e-7)
    result = model(torch.rand(2, 1, 4, 4))
    assert result.shape == (2, 10)
    assert torch.all(torch.isfinite(result))
