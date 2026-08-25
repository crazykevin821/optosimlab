import pytest
import torch

from optosimlab import (
    ElectroOpticComb,
    LinearDispersiveFiber,
    OverlapSaveFIR,
    PracticalElectroOpticComb,
    SimulationGrid,
    SmallSignalEDFA,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
def test_core_optical_modules_match_between_cpu_and_cuda() -> None:
    grid = SimulationGrid(128.0)
    field = torch.randn(2, 128, dtype=torch.complex64)
    modules = [
        LinearDispersiveFiber(grid, 10.0, attenuation_db_per_km=0.2, beta2_s2_per_m=-2e-26),
        ElectroOpticComb(grid, 8.0, modulation_indices_rad=0.7, rf_phases_rad=0.2),
        PracticalElectroOpticComb(grid, 8.0, modulation_indices_rad=0.7, rf_bandwidth_hz=16.0, enable_phase_noise=False),
        SmallSignalEDFA(grid, pump_current_ma=60.0, transparency_current_ma=20.0, gain_slope_db_per_ma=0.5, gain_bandwidth_hz=64.0),
        OverlapSaveFIR(torch.tensor([1 + 0j, 0.2 - 0.1j, -0.05j]), fft_size=16),
    ]
    for module in modules:
        cpu_output = module(field)
        cuda_output = module.cuda()(field.cuda()).cpu()
        assert torch.allclose(cuda_output, cpu_output, atol=2e-5, rtol=2e-5)
