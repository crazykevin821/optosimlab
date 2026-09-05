import torch

from optosimlab.units import db_to_field_ratio, db_to_power_ratio, dbm_to_w, field_from_power_w, w_to_dbm


def test_db_and_dbm_conversions_use_power_convention() -> None:
    assert torch.allclose(dbm_to_w(torch.tensor(0.0)), torch.tensor(1e-3))
    assert torch.allclose(w_to_dbm(torch.tensor(1e-3)), torch.tensor(0.0))
    assert torch.allclose(db_to_power_ratio(torch.tensor(20.0)), torch.tensor(100.0))
    assert torch.allclose(db_to_field_ratio(torch.tensor(20.0)), torch.tensor(10.0))


def test_field_squared_is_power_in_watts() -> None:
    field = field_from_power_w(torch.tensor([1e-3, 4e-3]), torch.tensor([0.0, 0.5]))
    assert torch.allclose(field.abs().square(), torch.tensor([1e-3, 4e-3]), rtol=1e-6)
