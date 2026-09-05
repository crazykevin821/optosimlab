import torch
import pytest

from optosimlab import MeasuredMZM


def test_measured_mzm_interpolates_power_and_clamps_voltage() -> None:
    mzm = MeasuredMZM(torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([0.0, 0.25, 1.0]))
    voltage = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    field = mzm(torch.ones(4, dtype=torch.complex64), voltage)
    assert torch.allclose(field.abs().square(), torch.tensor([0.0, 0.125, 0.625, 1.0]), atol=1e-6)


def test_measured_mzm_loads_named_csv_and_curve_has_gradient(tmp_path) -> None:
    path = tmp_path / "curve.csv"
    path.write_text("voltage_v,power_transmission,phase_rad\n-1,0.1,0\n0,0.5,0.2\n1,0.9,0.4\n", encoding="utf-8")
    mzm = MeasuredMZM.from_csv(path, phase_column="phase_rad", trainable_curve=True)
    result = mzm(torch.ones(3, dtype=torch.complex64), torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(result.abs().square(), torch.tensor([0.1, 0.5, 0.9]), atol=1e-6)
    result.real.sum().backward()
    assert mzm.power_transmission.grad is not None


def test_measured_mzm_rejects_nonmonotonic_voltage_points() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        MeasuredMZM(torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.1, 0.5, 0.9]))
