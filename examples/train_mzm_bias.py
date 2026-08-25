"""Optimise an MZM bias point through PyTorch autograd."""

import torch

from optosimlab import MachZehnderModulator, mean_power


def main() -> None:
    torch.manual_seed(0)
    samples = 256
    carrier = torch.ones(samples, dtype=torch.complex64)
    voltage = torch.zeros(samples)
    mzm = MachZehnderModulator(v_pi=1.0, bias_voltage=0.15, trainable=True)
    # This minimal example optimises just the bias. Other physical parameters
    # remain registered in the module but are intentionally fixed here.
    optimizer = torch.optim.Adam([mzm.bias_voltage], lr=0.03)
    target_power = torch.tensor(0.5)

    for step in range(200):
        optimizer.zero_grad()
        loss = (mean_power(mzm(carrier, voltage)) - target_power).square()
        loss.backward()
        optimizer.step()
        if step in {0, 49, 99, 199}:
            print(f"step={step:3d} bias={mzm.bias_voltage.item():+.5f} V  loss={loss.item():.3e}")

    assert loss.item() < 1e-5, "bias optimisation did not converge"


if __name__ == "__main__":
    main()
