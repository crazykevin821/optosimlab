# Changelog

## 0.9.0 - 2026-09-07

- Rebuilt `PhotonicRealConvolution` from the actual `WaveShaper`, MZM, `LinearDispersiveFiber`, and square-law photodetector device models; removed the duplicate elementary-math stand-ins.
- Unified splitter, directional coupler, ideal/practical/measured MZM, differential dual-port MZM, and four-optical-port MZM in `devices/mzm.py`; removed the overlapping `mzi.py` implementation.
- Added calibrated MZM inverse transfer functions so normalized input power is converted to physical drive voltage before modulation.
- Added explicit WDM control-frequency response to `WaveShaper` and carrier-dependent group delay to `LinearDispersiveFiber`.
- Added `GraphDevice` port wiring and `OpticalChain.from_device_graph()` for topology discovery, fan-out, merge, optional open ports, topological sorting, and cycle rejection.
- Added `PhotodetectorSum` and `BalancedPhotodetector` hardware definitions.
- Expanded physical, mathematical, graph, gradient, and regression verification to 91 passing tests plus one conditional CUDA test.

## 0.8.0 - 2026-09-06

- Defined SI-based optical field, power, voltage, time, frequency and dB conventions.
- Added commercial-style `GainControlledEDFA` with gain control, output-referenced saturation and ASE.
- Added trainable 1x2 splitter, unitary 2x2 directional coupler, SISO MZI and dual-input/single-output MZI.
- Extended `OpticalChain` from serial-only composition to named feed-forward graphs with fan-out and merge.
- Added `MeasuredMZM` with CSV loading, interpolation and calibrated-range clamping.
- Added Xu et al. Nature 2021 WDM real photonic convolution and differential signed convolution.
- Retained v0.7 current-controlled EDFA, four-input combiner and mathematical complex convolution as compatibility APIs.
- Expanded verification to 82 passing tests plus one conditional CUDA test.
