# Changelog

## 0.10.1 - 2026-10-26

- Replaced clipped positive/negative kernel splitting with balanced differential encoding so both physical branches retain useful gradients and trainable weights can cross zero.
- Added generator-based `iter_chunked()` execution for real, differential and MZI-complex convolution, allowing per-chunk backward passes without retaining every chunk graph.
- Projected all trainable intensity-MZI parameters and passive MZI complex-weight rows back into their physically admissible domains after optimizer updates.
- Corrected public API documentation parameter names and completed Chinese source documentation for the Issue #3 modules.
- Added real-MNIST acceptance runs for ideal, explicitly nonideal and intensity-MZI activation paths, plus long-record streaming-gradient validation.
- Regenerated the LaTeX manual, isolated wheel, and full verification record for Issue #3.

## 0.10.0 - 2026-10-18

- Added exact chunked execution for long real and complex one-dimensional sequences while preserving causal history and PyTorch autograd.
- Verified arbitrary leading batch dimensions for physical WDM real convolution and added full-output/full-gradient regression comparisons.
- Added `PhotonicMNISTClassifier`: one trainable signed optical convolution, selectable electronic ReLU or intensity-MZI activation, and one electronic linear layer.
- Added explicit ideal and nonideal MNIST modes. The nonideal path defines 3 dB MZM insertion loss, 20 dB extinction ratio, 0.2 dB/km fibre loss, and detector readout noise in watts.
- Added standalone unsigned `IntensityMZM` and `IntensityMZMActivation` models instead of deriving the intensity transfer from the differential MZM implementation.
- Added `MZIComplexWeightBank` and `MZIComplexConv1d`, using coherent complex weights parameterized by MZI internal/external phase shifts following Nature Communications 12, 457 (2021).
- Added an MNIST training script with a dependency-free IDX reader, synthetic smoke mode, physical-parameter projection, subset controls, and CPU/CUDA selection.
- Reworked documentation for portable relative commands, Chinese API explanations, parameter tables, model assumptions, and runnable examples.
- Expanded verification to 105 passing tests plus one conditional CUDA test.

## 0.9.1 - 2026-10-13

- Fixed the calibrated WDM real-convolution output for user-supplied dispersive fibre with nonzero attenuation.
- Added `LinearDispersiveFiber.power_transmission()` as the single differentiable definition of fibre power loss.
- Corrected the convolution leakage-floor subtraction and modulation-swing normalization to include common fibre transmission.
- Added an analytic regression test combining nonzero fibre loss with a finite-extinction practical MZM; the physical detector power and normalized convolution are both checked.
- Refreshed the LaTeX manual, formula self-check record, package instructions, and release version.

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
