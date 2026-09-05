# Changelog

## 0.8.0 - 2026-09-06

- Defined SI-based optical field, power, voltage, time, frequency and dB conventions.
- Added commercial-style `GainControlledEDFA` with gain control, output-referenced saturation and ASE.
- Added trainable 1x2 splitter, unitary 2x2 directional coupler, SISO MZI and dual-input/single-output MZI.
- Extended `OpticalChain` from serial-only composition to named feed-forward graphs with fan-out and merge.
- Added `MeasuredMZM` with CSV loading, interpolation and calibrated-range clamping.
- Added Xu et al. Nature 2021 WDM real photonic convolution and differential signed convolution.
- Retained v0.7 current-controlled EDFA, four-input combiner and mathematical complex convolution as compatibility APIs.
- Expanded verification to 82 passing tests plus one conditional CUDA test.
