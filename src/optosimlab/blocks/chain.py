"""Composable blocks for optical processing networks."""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn


class OpticalChain(nn.Module):
    """Sequentially apply modules that accept and return a field tensor."""

    def __init__(self, *devices: nn.Module | Iterable[nn.Module]) -> None:
        super().__init__()
        if len(devices) == 1 and not isinstance(devices[0], nn.Module):
            devices = tuple(devices[0])  # type: ignore[assignment]
        if not all(isinstance(device, nn.Module) for device in devices):
            raise TypeError("all OpticalChain entries must be torch.nn.Module instances")
        self.devices = nn.ModuleList(devices)  # type: ignore[arg-type]

    def forward(self, field: Tensor) -> Tensor:
        for device in self.devices:
            field = device(field)
        return field
