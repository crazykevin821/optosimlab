"""Real-valued WDM photonic convolution following Xu et al., Nature 2021."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SimulationGrid
from ..devices.base import require_complex_field
from ..devices.comb import OpticalFrequencyComb
from .chain import OpticalChain


class WaveShaperWeightBank(nn.Module):
    """Map nonnegative convolution weights to per-line optical power.

    Each channel field is multiplied by sqrt(w_k), so its power is multiplied
    by w_k.  A sigmoid parameterization keeps trainable weights inside (0, 1).
    """

    def __init__(self, power_weights: Tensor, *, trainable: bool = True) -> None:
        super().__init__()
        weights = torch.as_tensor(power_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() < 1:
            raise ValueError("power_weights must be a non-empty one-dimensional tensor")
        if torch.any((weights < 0) | (weights > 1)):
            raise ValueError("power_weights must lie in [0, 1]")
        epsilon = torch.finfo(weights.dtype).eps
        self.weight_logits = nn.Parameter(torch.logit(weights.clamp(epsilon, 1.0 - epsilon)), requires_grad=trainable)

    @property
    def power_weights(self) -> Tensor:
        return torch.sigmoid(self.weight_logits)

    def forward(self, channel_fields: Tensor) -> Tensor:
        require_complex_field(channel_fields)
        if channel_fields.ndim < 2 or channel_fields.shape[-2] != self.weight_logits.numel():
            raise ValueError("channel_fields must have shape (..., channels, samples) matching the weights")
        weights = self.power_weights.to(device=channel_fields.device, dtype=channel_fields.real.dtype)
        shape = [1] * channel_fields.ndim
        shape[-2] = weights.numel()
        return channel_fields * torch.sqrt(weights).view(shape)


class IntensityBroadcastModulator(nn.Module):
    """Broadcast a nonnegative normalized sample stream onto every WDM line.

    Multiplication by sqrt(x[n]) makes each modulated channel power
    proportional to x[n].  This is the calibrated intensity-domain abstraction
    of the single MZM used to broadcast the input waveform.
    """

    def forward(self, channel_fields: Tensor, samples: Tensor) -> Tensor:
        require_complex_field(channel_fields)
        if not isinstance(samples, Tensor) or torch.is_complex(samples) or not samples.is_floating_point():
            raise TypeError("samples must be a real floating-point torch.Tensor")
        if samples.shape[-1] != channel_fields.shape[-1]:
            raise ValueError("samples and channel_fields must share their final dimension")
        if torch.any(samples.detach() < 0):
            raise ValueError("single-branch intensity convolution requires nonnegative input samples")
        samples = samples.to(device=channel_fields.device, dtype=channel_fields.real.dtype)
        while samples.ndim < channel_fields.ndim:
            samples = samples.unsqueeze(-2)
        return channel_fields * torch.sqrt(samples)


class ProgressiveDelay(nn.Module):
    """Apply zero-filled delay k*tap_delay_samples to wavelength channel k."""

    def __init__(self, tap_delay_samples: int = 1) -> None:
        super().__init__()
        if not isinstance(tap_delay_samples, int) or tap_delay_samples < 1:
            raise ValueError("tap_delay_samples must be a positive integer")
        self.tap_delay_samples = tap_delay_samples

    def forward(self, channel_fields: Tensor) -> Tensor:
        require_complex_field(channel_fields)
        if channel_fields.ndim < 2:
            raise ValueError("channel_fields must have shape (..., channels, samples)")
        channels = channel_fields.shape[-2]
        samples = channel_fields.shape[-1]
        delayed = torch.zeros_like(channel_fields)
        for channel in range(channels):
            shift = channel * self.tap_delay_samples
            if shift < samples:
                delayed[..., channel, shift:] = channel_fields[..., channel, : samples - shift]
        return delayed


class PhotodetectorSum(nn.Module):
    """Square-law detect WDM channels and sum their powers in watts.

    The sum-of-channel-powers model assumes the receiver rejects beat notes at
    the comb spacing, as in a detector/electrical bandwidth below that spacing.
    """

    def forward(self, channel_fields: Tensor) -> Tensor:
        require_complex_field(channel_fields)
        if channel_fields.ndim < 2:
            raise ValueError("channel_fields must have shape (..., channels, samples)")
        return channel_fields.abs().square().sum(dim=-2)


class PhotonicRealConvolution(nn.Module):
    """End-to-end nonnegative real convolution built as an OpticalChain graph.

    The physical detector output is

        P_det[n] = P_line * sum_k w[k] x[n-kD]

    for equal comb-line power P_line and tap delay D.  forward returns the
    calibrated dimensionless convolution P_det/P_line, while
    detected_power_w returns the physical detector power in W.
    """

    def __init__(
        self,
        power_weights: Tensor,
        *,
        sample_rate_hz: float = 100e9,
        line_spacing_hz: float = 10e9,
        line_power_w: float = 1e-3,
        tap_delay_samples: int = 1,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(power_weights, dtype=torch.float32)
        if line_power_w <= 0:
            raise ValueError("line_power_w must be positive")
        self.line_power_w = float(line_power_w)
        self.grid = SimulationGrid(sample_rate_hz=sample_rate_hz)
        self.comb = OpticalFrequencyComb(
            self.grid,
            line_spacing_hz=line_spacing_hz,
            line_count=weights.numel(),
            line_powers=line_power_w,
            trainable=False,
        )
        weight_bank = WaveShaperWeightBank(weights, trainable=trainable)
        modulator = IntensityBroadcastModulator()
        delay = ProgressiveDelay(tap_delay_samples)
        detector = PhotodetectorSum()
        self.optical_chain = OpticalChain.from_graph(
            [
                ("weights", weight_bank, "comb", "weighted"),
                ("broadcast", modulator, ("weighted", "samples"), "modulated"),
                ("delay", delay, "modulated", "delayed"),
                ("detector", detector, "delayed", "power_w"),
            ],
            outputs="power_w",
        )

    @property
    def power_weights(self) -> Tensor:
        weight_bank = self.optical_chain.graph_devices["weights"]
        assert isinstance(weight_bank, WaveShaperWeightBank)
        return weight_bank.power_weights

    def detected_power_w(self, samples: Tensor) -> Tensor:
        """Return physical square-law detector output power in W."""
        if not isinstance(samples, Tensor) or torch.is_complex(samples) or not samples.is_floating_point():
            raise TypeError("samples must be a real floating-point torch.Tensor")
        channels = self.comb.channel_fields(
            samples.shape[-1], device=samples.device, real_dtype=samples.dtype
        )
        if samples.ndim > 1:
            channels = channels.view(*([1] * (samples.ndim - 1)), *channels.shape)
        return self.optical_chain({"comb": channels, "samples": samples})

    def forward(self, samples: Tensor) -> Tensor:
        """Return calibrated causal real convolution with zero initial state."""
        return self.detected_power_w(samples) / self.line_power_w


class DifferentialPhotonicConvolution(nn.Module):
    """Signed real convolution using positive and negative detector branches."""

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
