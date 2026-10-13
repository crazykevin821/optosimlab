"""Hardware-composed real WDM convolution after Xu et al., Nature 2021."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import SimulationGrid
from ..devices.comb import OpticalFrequencyComb
from ..devices.fiber import LinearDispersiveFiber
from ..devices.mzm import IdealMZM, MeasuredMZM, MachZehnderModulator
from ..devices.photodetector import PhotodetectorSum
from ..devices.waveshaper import WaveShaper
from .chain import OpticalChain


class PhotonicRealConvolution(nn.Module):
    """Nonnegative real convolution composed only from physical devices.

    The executed graph is OpticalFrequencyComb -> WaveShaper -> MZM ->
    LinearDispersiveFiber -> PhotodetectorSum.  The input sequence is converted
    to MZM drive voltage through the selected modulator's calibrated inverse.
    Fibre length is chosen from beta2 so adjacent comb lines acquire exactly
    tap_delay_samples of relative delay in the narrowband WDM approximation.
    """

    def __init__(
        self,
        power_weights: Tensor,
        *,
        sample_rate_hz: float = 100e9,
        line_spacing_hz: float = 10e9,
        line_power_w: float = 1e-3,
        tap_delay_samples: int = 1,
        dispersion_ps_nm_km: float = 17.0,
        modulator: IdealMZM | MachZehnderModulator | MeasuredMZM | None = None,
        fiber: LinearDispersiveFiber | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(power_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() < 1:
            raise ValueError("power_weights must be a non-empty one-dimensional tensor")
        if torch.any((weights < 0) | (weights > 1)):
            raise ValueError("power_weights must lie in [0, 1]")
        if line_power_w <= 0:
            raise ValueError("line_power_w must be positive")
        if not isinstance(tap_delay_samples, int) or tap_delay_samples < 1:
            raise ValueError("tap_delay_samples must be a positive integer")
        if line_spacing_hz <= 0:
            raise ValueError("line_spacing_hz must be positive")

        self.line_power_w = float(line_power_w)
        self.tap_delay_samples = tap_delay_samples
        self.grid = SimulationGrid(sample_rate_hz=sample_rate_hz)
        self.comb = OpticalFrequencyComb(
            self.grid,
            line_spacing_hz=line_spacing_hz,
            line_count=weights.numel(),
            line_powers=line_power_w,
            trainable=False,
        )
        offsets = self.comb.nominal_line_offsets_hz(dtype=torch.float32)
        attenuation_db = -10.0 * torch.log10(weights.clamp_min(1e-30))
        if weights.numel() == 1:
            control_frequencies = torch.tensor([-line_spacing_hz / 2.0, line_spacing_hz / 2.0])
            attenuation_db = attenuation_db.expand(2).clone()
        else:
            control_frequencies = offsets
        waveshaper = WaveShaper(
            self.grid,
            control_frequencies_hz=control_frequencies,
            attenuation_db=attenuation_db,
            trainable=trainable,
        )
        waveshaper.phase_rad.requires_grad_(False)

        selected_modulator = modulator if modulator is not None else IdealMZM(trainable=False)
        if not hasattr(selected_modulator, "voltage_for_power_transmission") or not hasattr(
            selected_modulator, "power_transmission_bounds"
        ):
            raise TypeError("modulator must provide inverse transfer and power-transmission bounds")

        if fiber is None:
            beta2 = LinearDispersiveFiber.beta2_from_dispersion(
                dispersion_ps_nm_km, self.grid.center_wavelength_m
            )
            if beta2 >= 0:
                raise ValueError("dispersion must produce negative beta2 for increasing-frequency positive delays")
            delay_s = tap_delay_samples / sample_rate_hz
            length_m = delay_s / (-beta2 * 2.0 * math.pi * line_spacing_hz)
            selected_fiber = LinearDispersiveFiber(
                self.grid,
                length_m=length_m,
                beta2_s2_per_m=beta2,
                attenuation_db_per_km=0.0,
                trainable=False,
            )
        else:
            selected_fiber = fiber
            actual_delay = float(selected_fiber.adjacent_channel_delay_s(line_spacing_hz).detach())
            expected_delay = tap_delay_samples / sample_rate_hz
            if not math.isclose(actual_delay, expected_delay, rel_tol=1e-5, abs_tol=1e-15):
                raise ValueError("provided fiber does not match tap_delay_samples at this line spacing")

        detector = PhotodetectorSum()
        self.optical_chain = OpticalChain.from_graph(
            [
                ("waveshaper", waveshaper, ("comb", "offsets"), "weighted"),
                ("modulator", selected_modulator, ("weighted", "voltage"), "modulated"),
                ("fiber", selected_fiber, ("modulated", "offsets"), "delayed"),
                ("photodetector", detector, "delayed", "power_w"),
            ],
            outputs="power_w",
        )

    @property
    def waveshaper(self) -> WaveShaper:
        module = self.optical_chain.graph_devices["waveshaper"]
        assert isinstance(module, WaveShaper)
        return module

    @property
    def modulator(self) -> IdealMZM | MachZehnderModulator | MeasuredMZM:
        module = self.optical_chain.graph_devices["modulator"]
        assert isinstance(module, (IdealMZM, MachZehnderModulator, MeasuredMZM))
        return module

    @property
    def fiber(self) -> LinearDispersiveFiber:
        module = self.optical_chain.graph_devices["fiber"]
        assert isinstance(module, LinearDispersiveFiber)
        return module

    @property
    def detector(self) -> PhotodetectorSum:
        module = self.optical_chain.graph_devices["photodetector"]
        assert isinstance(module, PhotodetectorSum)
        return module

    @property
    def power_weights(self) -> Tensor:
        offsets = self.comb.nominal_line_offsets_hz(
            device=self.waveshaper.attenuation_db.device,
            dtype=self.waveshaper.attenuation_db.dtype,
        )
        return self.waveshaper.power_transmission_at(offsets)

    def detected_power_w(self, samples: Tensor) -> Tensor:
        """Execute the physical device graph and return detector power in W."""
        if not isinstance(samples, Tensor) or torch.is_complex(samples) or not samples.is_floating_point():
            raise TypeError("samples must be a real floating-point torch.Tensor")
        if torch.any((samples.detach() < 0) | (samples.detach() > 1)):
            raise ValueError("MZM-normalized input samples must be nonnegative and at most 1")
        maximum_delay = (self.comb.line_count - 1) * self.tap_delay_samples
        padded_samples = F.pad(samples, (0, maximum_delay))
        total_samples = padded_samples.shape[-1]
        offsets = self.comb.nominal_line_offsets_hz(device=samples.device, dtype=samples.dtype)
        channels = self.comb.channel_envelopes(
            total_samples, device=samples.device, real_dtype=samples.dtype
        )
        if samples.ndim > 1:
            channels = channels.view(*([1] * (samples.ndim - 1)), *channels.shape)
        minimum, maximum = self.modulator.power_transmission_bounds(padded_samples)
        swing = maximum - minimum
        if torch.any(swing.detach() <= torch.finfo(samples.dtype).eps):
            raise ValueError("modulator must have nonzero reachable power-transmission swing")
        target_transmission = minimum + swing * padded_samples
        voltage = self.modulator.voltage_for_power_transmission(target_transmission)
        while voltage.ndim < channels.ndim:
            voltage = voltage.unsqueeze(-2)
        power = self.optical_chain({"comb": channels, "offsets": offsets, "voltage": voltage})
        return power[..., : samples.shape[-1]]

    def forward(self, samples: Tensor) -> Tensor:
        """Return calibrated zero-state causal convolution."""
        detected = self.detected_power_w(samples)
        minimum, maximum = self.modulator.power_transmission_bounds(samples)
        fiber_transmission = self.fiber.power_transmission(samples)
        weighted_line_power = self.line_power_w * self.power_weights.sum().to(
            device=samples.device, dtype=samples.dtype
        )
        baseline = fiber_transmission * minimum * weighted_line_power
        calibration = self.line_power_w * fiber_transmission * (maximum - minimum)
        if torch.any(calibration.detach() <= torch.finfo(samples.dtype).tiny):
            raise ValueError("combined line power, fibre transmission and MZM swing are too small to calibrate")
        return (detected - baseline) / calibration


class DifferentialPhotonicConvolution(nn.Module):
    """Signed convolution made from two physical convolution paths."""

    def __init__(self, weights: Tensor, **kwargs: object) -> None:
        super().__init__()
        values = torch.as_tensor(weights, dtype=torch.float32)
        if values.ndim != 1 or values.numel() < 1:
            raise ValueError("weights must be a non-empty one-dimensional tensor")
        scale = values.abs().max().clamp_min(torch.finfo(values.dtype).eps)
        self.register_buffer("weight_scale", scale)
        self.positive = PhotonicRealConvolution((values / scale).clamp_min(0.0), **kwargs)
        self.negative = PhotonicRealConvolution((-values / scale).clamp_min(0.0), **kwargs)

    def forward(self, samples: Tensor) -> Tensor:
        return self.weight_scale * (self.positive(samples) - self.negative(samples))
