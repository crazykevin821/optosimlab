import math

import torch

from optosimlab.blocks import OpticalChain
from optosimlab.devices import OpticalAttenuator
from optosimlab.metrics import mean_power, normalized_mean_square_error, snr_db


def test_optical_chain_composes_devices_in_order() -> None:
    field = torch.ones(64, dtype=torch.complex64)
    chain = OpticalChain(OpticalAttenuator(3.0), OpticalAttenuator(3.0))
    assert torch.allclose(mean_power(chain(field)), torch.tensor(10 ** (-0.6)), rtol=1e-5)


def test_metrics_match_known_power_ratios() -> None:
    signal = torch.ones(32, dtype=torch.complex64)
    noise = torch.full((32,), 0.1 + 0.0j, dtype=torch.complex64)
    assert torch.allclose(normalized_mean_square_error(signal, signal), torch.tensor(0.0))
    assert math.isclose(snr_db(signal, noise).item(), 20.0, rel_tol=1e-5)
