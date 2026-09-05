"""Sequential and directed-acyclic optical processing networks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch import Tensor, nn


class OpticalChain(nn.Module):
    """Compose serial devices or an explicitly wired feed-forward graph.

    The ordinary constructor preserves the v0.7 sequential API.  Use
    OpticalChain.from_graph for fan-out, parallel branches, multiple inputs and
    multiple outputs.  Graph nodes are executed in the supplied topological
    order and have the form (name, module, input_names, output_names).
    """

    def __init__(self, *devices: nn.Module | Iterable[nn.Module]) -> None:
        super().__init__()
        if len(devices) == 1 and not isinstance(devices[0], nn.Module):
            devices = tuple(devices[0])  # type: ignore[assignment]
        if not all(isinstance(device, nn.Module) for device in devices):
            raise TypeError("all OpticalChain entries must be torch.nn.Module instances")
        self.devices = nn.ModuleList(devices)  # type: ignore[arg-type]
        self.graph_devices = nn.ModuleDict()
        self._graph_specs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self._graph_outputs: tuple[str, ...] = ()
        self._is_graph = False

    @staticmethod
    def _names(value: str | Sequence[str], label: str) -> tuple[str, ...]:
        names = (value,) if isinstance(value, str) else tuple(value)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"{label} must contain at least one non-empty string")
        return names

    @classmethod
    def from_graph(
        cls,
        nodes: Sequence[tuple[str, nn.Module, str | Sequence[str], str | Sequence[str]]],
        outputs: str | Sequence[str],
    ) -> "OpticalChain":
        """Build a graph from nodes already listed in topological order."""
        chain = cls()
        seen_nodes: set[str] = set()
        produced_values: set[str] = set()
        for node_name, module, input_names, output_names in nodes:
            if not isinstance(module, nn.Module):
                raise TypeError(f"graph node {node_name!r} is not a torch.nn.Module")
            if not node_name or "." in node_name or node_name in seen_nodes:
                raise ValueError("graph node names must be unique, non-empty and contain no dots")
            inputs_tuple = cls._names(input_names, "input_names")
            outputs_tuple = cls._names(output_names, "output_names")
            duplicate = produced_values.intersection(outputs_tuple)
            if duplicate:
                raise ValueError(f"graph values are produced more than once: {sorted(duplicate)}")
            chain.graph_devices[node_name] = module
            chain._graph_specs.append((node_name, inputs_tuple, outputs_tuple))
            seen_nodes.add(node_name)
            produced_values.update(outputs_tuple)
        chain._graph_outputs = cls._names(outputs, "outputs")
        chain._is_graph = True
        return chain

    def _forward_graph(self, inputs: Tensor | Mapping[str, Any]) -> Any:
        values: dict[str, Any] = {"input": inputs} if isinstance(inputs, Tensor) else dict(inputs)
        for node_name, input_names, output_names in self._graph_specs:
            missing = [name for name in input_names if name not in values]
            if missing:
                raise KeyError(f"graph node {node_name!r} is missing inputs {missing}")
            result = self.graph_devices[node_name](*(values[name] for name in input_names))
            if len(output_names) == 1:
                values[output_names[0]] = result
                continue
            if not isinstance(result, (tuple, list)) or len(result) != len(output_names):
                raise ValueError(
                    f"graph node {node_name!r} declared {len(output_names)} outputs "
                    "but did not return a matching tuple/list"
                )
            values.update(zip(output_names, result))
        missing_outputs = [name for name in self._graph_outputs if name not in values]
        if missing_outputs:
            raise KeyError(f"graph outputs were never produced: {missing_outputs}")
        if len(self._graph_outputs) == 1:
            return values[self._graph_outputs[0]]
        return {name: values[name] for name in self._graph_outputs}

    def forward(self, field: Tensor | Mapping[str, Any]) -> Any:
        if self._is_graph:
            return self._forward_graph(field)
        if not isinstance(field, Tensor):
            raise TypeError("sequential OpticalChain input must be a torch.Tensor")
        for device in self.devices:
            field = device(field)
        return field
