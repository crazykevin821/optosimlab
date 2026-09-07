"""Run a physical-device WDM convolution and print the registered hardware path."""

import torch

from optosimlab import PhotonicRealConvolution


def main() -> None:
    samples = torch.tensor([0.1, 0.3, 0.7, 1.0, 0.2, 0.6])
    weights = torch.tensor([0.2, 0.5, 0.8])
    layer = PhotonicRealConvolution(weights, tap_delay_samples=1, trainable=False)
    print("calibrated convolution:", layer(samples))
    print("detector power (W):", layer.detected_power_w(samples))
    print("physical graph:", tuple(layer.optical_chain.graph_devices))


if __name__ == "__main__":
    main()
