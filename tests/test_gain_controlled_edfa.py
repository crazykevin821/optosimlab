import torch

from optosimlab import GainControlledEDFA, SimulationGrid


def test_gain_controlled_edfa_peak_and_fwhm_are_power_correct() -> None:
    edfa = GainControlledEDFA(
        SimulationGrid(64.0), gain_db=20.0, gain_bandwidth_hz=16.0,
        enable_saturation=False, enable_ase=False, trainable=False,
    )
    gain = edfa.small_signal_power_gain(64)
    assert torch.allclose(gain[0], torch.tensor(100.0), rtol=1e-6)
    assert torch.allclose(gain[8], torch.tensor(50.0), rtol=1e-6)


def test_gain_control_is_not_pump_current_and_saturation_reduces_gain() -> None:
    edfa = GainControlledEDFA(
        SimulationGrid(64.0), gain_db=20.0, gain_bandwidth_hz=64.0,
        output_saturation_power_dbm=0.0, enable_ase=False, trainable=False,
    )
    assert not hasattr(edfa, "pump_current_ma")
    weak = torch.full((64,), 1e-5**0.5, dtype=torch.complex64)
    strong = torch.full((64,), 1e-2**0.5, dtype=torch.complex64)
    assert edfa.effective_power_gain(strong)[0] < edfa.effective_power_gain(weak)[0]


def test_gain_controlled_edfa_is_differentiable_and_ase_psd_nonnegative() -> None:
    edfa = GainControlledEDFA(SimulationGrid(64.0), enable_ase=False)
    field = torch.full((64,), 1e-4**0.5, dtype=torch.complex64)
    loss = edfa(field).abs().square().mean()
    loss.backward()
    assert edfa.gain_db.grad is not None and edfa.gain_db.grad.abs() > 0
    gain = edfa.effective_power_gain(field)
    assert torch.all(edfa.ase_power_spectral_density(gain) >= 0)
