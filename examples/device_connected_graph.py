"""Construct a branched optical system directly from device port connections."""

import torch

from optosimlab import DirectionalCoupler2x2, OpticalAttenuator, OpticalChain, PowerSplitter1x2


def main() -> None:
    splitter = PowerSplitter1x2(0.5, trainable=False).named("splitter")
    upper = OpticalAttenuator(3.0, trainable=False).named("upper")
    lower = OpticalAttenuator(6.0, trainable=False).named("lower")
    combiner = DirectionalCoupler2x2(0.5, trainable=False).named("combiner")
    splitter.connect_outputs(upper, lower)
    combiner.connect_inputs(upper, lower)
    chain = OpticalChain.from_device_graph(splitter, outputs=[(combiner, "output0")])
    result = chain(torch.ones(16, dtype=torch.complex64))
    print("output mean power (W):", result.abs().square().mean().item())


if __name__ == "__main__":
    main()
