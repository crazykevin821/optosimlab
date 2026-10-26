"""光学器件共享的内部校验与图连接工具。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphDevice(nn.Module):
    """能够接入 OpticalChain 图的器件基类。

    连接仅作为拓扑元数据保存，刻意不写入 PyTorch 子模块注册表。
    ``OpticalChain.from_device_graph`` 会且只会注册每个可达器件一次。
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
        """设置稳定的图节点名称，并返回自身以支持链式构建。"""
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
        """把一个输出端口连接到目标器件的一个输入端口。

        允许传入 ``None``，以便让可选分支保持未连接状态。
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
        """按顺序连接输出端口；省略或为 None 的目标保持悬空。"""
        if len(targets) > len(self.graph_output_ports):
            raise ValueError("too many targets for this device")
        for output_index, target in enumerate(targets):
            self.connect(target, output_port=output_index, input_port=0)
        return self

    def connect_inputs(
        self,
        *sources: "GraphDevice | tuple[GraphDevice, int | str] | None",
    ) -> "GraphDevice":
        """按顺序从器件或 ``(器件, 输出端口)`` 连接输入端口。"""
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
    """在模块边界拒绝语义含糊的实数光场。"""
    if not isinstance(field, Tensor):
        raise TypeError("field must be a torch.Tensor")
    if not torch.is_complex(field):
        raise TypeError("field must use a complex dtype (complex64 or complex128)")
    if field.ndim < 1:
        raise ValueError("field must contain a final sample dimension")


def nonnegative_parameter(value: Tensor, name: str, *, floor: float = 0.0) -> Tensor:
    """在保留 autograd 连接的同时校验非负物理参数。"""
    if torch.any(value.detach() < 0):
        raise ValueError(f"{name} must not be negative")
    return value.clamp_min(floor)
