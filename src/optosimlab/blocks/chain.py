"""串行与有向无环图形式的光学处理网络。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from ..devices.base import GraphDevice


class OpticalChain(nn.Module):
    """组合串行器件或显式连线的前馈图。

    普通构造器保留 v0.7 的串行 API。需要扇出、并行分支、多输入或多输出时，
    使用 ``OpticalChain.from_graph``。图节点按给定拓扑顺序执行，节点格式为
    ``(名称, 模块, 输入名, 输出名)``。
    """

    def __init__(self, *devices: nn.Module | Iterable[nn.Module]) -> None:
        super().__init__()
        if len(devices) == 1 and not isinstance(devices[0], nn.Module):
            devices = tuple(devices[0])  # 类型检查器无法推断运行时展平结果
        if not all(isinstance(device, nn.Module) for device in devices):
            raise TypeError("all OpticalChain entries must be torch.nn.Module instances")
        self.devices = nn.ModuleList(devices)  # 已在上方校验全部元素为 nn.Module
        self.graph_devices = nn.ModuleDict()
        self._graph_specs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self._graph_outputs: tuple[str, ...] = ()
        self._is_graph = False
        self.connected_devices = nn.ModuleDict()
        self._connected_specs: list[tuple[str, tuple[tuple[int, int] | None, ...]]] = []
        self._connected_outputs: list[tuple[int, int, str]] = []
        self._is_device_graph = False

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
        """根据已按拓扑顺序排列的节点构建图。"""
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

    @classmethod
    def from_device_graph(
        cls,
        devices: GraphDevice | Sequence[GraphDevice],
        *,
        outputs: Sequence[GraphDevice | tuple[GraphDevice, int | str]] | None = None,
    ) -> "OpticalChain":
        """沿器件上声明的连接自动构建链路图。

        连通分量中的任一器件都可作为种子。本方法遍历其上下游、检查环路、
        仅注册每个模块一次，并自动发现未连接的图输入与图输出。
        """
        seeds = (devices,) if isinstance(devices, GraphDevice) else tuple(devices)
        if not seeds or not all(isinstance(device, GraphDevice) for device in seeds):
            raise TypeError("devices must contain at least one GraphDevice")

        discovered: list[GraphDevice] = []
        seen: set[int] = set()
        pending = list(seeds)
        while pending:
            device = pending.pop(0)
            if id(device) in seen:
                continue
            seen.add(id(device))
            discovered.append(device)
            pending.extend(source for source, _ in device.graph_upstream.values())
            for connections in device.graph_downstream.values():
                pending.extend(target for target, _ in connections)

        node_set = {id(device) for device in discovered}
        indegree = {
            id(device): sum(1 for source, _ in device.graph_upstream.values() if id(source) in node_set)
            for device in discovered
        }
        queue = [device for device in discovered if indegree[id(device)] == 0]
        ordered: list[GraphDevice] = []
        while queue:
            device = queue.pop(0)
            ordered.append(device)
            for connections in device.graph_downstream.values():
                for target, _ in connections:
                    if id(target) not in node_set:
                        continue
                    indegree[id(target)] -= 1
                    if indegree[id(target)] == 0:
                        queue.append(target)
        if len(ordered) != len(discovered):
            raise ValueError("device graph contains a cycle")

        chain = cls()
        index_by_id = {id(device): index for index, device in enumerate(ordered)}
        used_names: set[str] = set()
        names: list[str] = []
        for device in ordered:
            base = device.graph_name
            name = base
            suffix = 2
            while name in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(name)
            names.append(name)
            chain.connected_devices[name] = device

        for device, name in zip(ordered, names):
            incoming: list[tuple[int, int] | None] = []
            for input_index in range(len(device.graph_input_ports)):
                edge = device.graph_upstream.get(input_index)
                incoming.append(None if edge is None else (index_by_id[id(edge[0])], edge[1]))
            chain._connected_specs.append((name, tuple(incoming)))

        if outputs is None:
            selected = []
            for device in ordered:
                for output_index in range(len(device.graph_output_ports)):
                    if not device.graph_downstream.get(output_index):
                        selected.append((device, output_index))
        else:
            selected = []
            for item in outputs:
                if isinstance(item, tuple):
                    device, port = item
                else:
                    device, port = item, 0
                if id(device) not in index_by_id:
                    raise ValueError("an explicitly selected output device is not in the connected graph")
                selected.append((device, device._port_index(port, output=True)))
        if not selected:
            raise ValueError("device graph has no selected or unconnected outputs")
        chain._connected_outputs = [
            (index_by_id[id(device)], port, f"{names[index_by_id[id(device)]]}.{device.graph_output_ports[port]}")
            for device, port in selected
        ]
        chain._is_device_graph = True
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

    def _forward_device_graph(self, inputs: Any) -> Any:
        external_slots: list[tuple[int, int, str, str]] = []
        for device_index, (name, incoming) in enumerate(self._connected_specs):
            device = self.connected_devices[name]
            assert isinstance(device, GraphDevice)
            for input_index, edge in enumerate(incoming):
                if edge is None:
                    port = device.graph_input_ports[input_index]
                    external_slots.append((device_index, input_index, name, port))

        external: dict[tuple[int, int], Any] = {}
        if isinstance(inputs, Mapping):
            for device_index, input_index, name, port in external_slots:
                qualified = f"{name}.{port}"
                if qualified in inputs:
                    external[(device_index, input_index)] = inputs[qualified]
                elif len(self.connected_devices[name].graph_input_ports) == 1 and name in inputs:
                    external[(device_index, input_index)] = inputs[name]
                elif len(external_slots) == 1 and "input" in inputs:
                    external[(device_index, input_index)] = inputs["input"]
                else:
                    raise KeyError(f"missing external device input {qualified!r}")
        else:
            if len(external_slots) != 1:
                names = [f"{name}.{port}" for _, _, name, port in external_slots]
                raise ValueError(f"device graph requires a mapping for external inputs {names}")
            external[(external_slots[0][0], external_slots[0][1])] = inputs

        node_outputs: list[tuple[Any, ...]] = []
        for device_index, (name, incoming) in enumerate(self._connected_specs):
            arguments = []
            for input_index, edge in enumerate(incoming):
                arguments.append(external[(device_index, input_index)] if edge is None else node_outputs[edge[0]][edge[1]])
            device = self.connected_devices[name]
            result = device(*arguments)
            output_count = len(device.graph_output_ports)
            if output_count == 1:
                normalized = (result,)
            elif isinstance(result, (tuple, list)) and len(result) == output_count:
                normalized = tuple(result)
            else:
                raise ValueError(f"device {name!r} must return {output_count} graph outputs")
            node_outputs.append(normalized)

        result_map = {label: node_outputs[index][port] for index, port, label in self._connected_outputs}
        return next(iter(result_map.values())) if len(result_map) == 1 else result_map

    def forward(self, field: Tensor | Mapping[str, Any]) -> Any:
        if self._is_device_graph:
            return self._forward_device_graph(field)
        if self._is_graph:
            return self._forward_graph(field)
        if not isinstance(field, Tensor):
            raise TypeError("sequential OpticalChain input must be a torch.Tensor")
        for device in self.devices:
            field = device(field)
        return field
