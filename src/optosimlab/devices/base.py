"""Internal validation helpers shared by optical devices."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphDevice(nn.Module):
    """Base class for devices that can be wired into an OpticalChain graph.

    Connections are topology metadata only and are deliberately stored outside
    PyTorch's child-module registry.  OpticalChain.from_device_graph registers
    each reachable device exactly once.
    """

    graph_input_ports: tuple[str, ...] = ("input",)
    graph_output_ports: tuple[str, ...] = ("output",)

    def _port_index(self, port: int | str, *, output: bool) -> int:
        ports = self.graph_output_ports if output else self.graph_input_ports
        if isinstance(port, str):
            if port not in ports:
                raise ValueError(f"unknown {'output' if output else 'input'} port {port!r}; expected one of {ports}")
            return ports.index(port)
        if not isinstance(port, int) or port < 0 or port >= len(ports):
            raise ValueError(f"port index must be in [0, {len(ports) - 1}]")
        return port

    @property
    def graph_downstream(self) -> dict[int, list[tuple["GraphDevice", int]]]:
        return self.__dict__.setdefault("_graph_downstream", {})

    @property
    def graph_upstream(self) -> dict[int, tuple["GraphDevice", int]]:
        return self.__dict__.setdefault("_graph_upstream", {})

    @property
    def graph_name(self) -> str:
        return self.__dict__.get("_graph_name", self.__class__.__name__)

    def named(self, name: str) -> "GraphDevice":
        """Assign a stable graph name and return self for fluent construction."""
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("graph name must be non-empty and contain no dots")
        self.__dict__["_graph_name"] = name
        return self

    def connect(
        self,
        target: "GraphDevice | None",
        *,
        output_port: int | str = 0,
        input_port: int | str = 0,
    ) -> "GraphDevice | None":
        """Connect one output port to one target input port.

        None is accepted so optional branches can be left unconnected.
        """
        if target is None:
            return None
        if not isinstance(target, GraphDevice):
            raise TypeError("target must inherit GraphDevice")
        source_index = self._port_index(output_port, output=True)
        target_index = target._port_index(input_port, output=False)
        if target_index in target.graph_upstream:
            raise ValueError(f"{target.graph_name}.{target.graph_input_ports[target_index]} already has an upstream device")
        edge = (target, target_index)
        if edge in self.graph_downstream.get(source_index, []):
            raise ValueError("this device connection already exists")
        self.graph_downstream.setdefault(source_index, []).append(edge)
        target.graph_upstream[target_index] = (self, source_index)
        return target

    def connect_outputs(self, *targets: "GraphDevice | None") -> "GraphDevice":
        """Connect output ports in order; omitted or None targets remain open."""
        if len(targets) > len(self.graph_output_ports):
            raise ValueError("too many targets for this device")
        for output_index, target in enumerate(targets):
            self.connect(target, output_port=output_index, input_port=0)
        return self

    def connect_inputs(
        self,
        *sources: "GraphDevice | tuple[GraphDevice, int | str] | None",
    ) -> "GraphDevice":
        """Connect input ports in order from devices or (device, output_port)."""
        if len(sources) > len(self.graph_input_ports):
            raise ValueError("too many sources for this device")
        for input_index, source in enumerate(sources):
            if source is None:
                continue
            if isinstance(source, tuple):
                source_device, output_port = source
            else:
                source_device, output_port = source, 0
            if not isinstance(source_device, GraphDevice):
                raise TypeError("source must inherit GraphDevice")
            source_device.connect(self, output_port=output_port, input_port=input_index)
        return self


def require_complex_field(field: Tensor) -> None:
    """Reject ambiguous real-valued optical fields at the module boundary."""
    if not isinstance(field, Tensor):
        raise TypeError("field must be a torch.Tensor")
    if not torch.is_complex(field):
        raise TypeError("field must use a complex dtype (complex64 or complex128)")
    if field.ndim < 1:
        raise ValueError("field must contain a final sample dimension")


def nonnegative_parameter(value: Tensor, name: str, *, floor: float = 0.0) -> Tensor:
    """Validate a physical parameter while preserving its autograd connection."""
    if torch.any(value.detach() < 0):
        raise ValueError(f"{name} must not be negative")
    return value.clamp_min(floor)
