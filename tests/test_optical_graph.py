import pytest
import torch
from torch import nn

from optosimlab import DirectionalCoupler2x2, OpticalAttenuator, OpticalChain, PowerSplitter1x2


class Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value * x


class Add(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


def test_graph_supports_fanout_parallel_paths_and_merge() -> None:
    graph = OpticalChain.from_graph(
        [
            ("split", PowerSplitter1x2(0.5, trainable=False), "input", ("a", "b")),
            ("left", Scale(2.0), "a", "c"),
            ("right", Scale(3.0), "b", "d"),
            ("sum", Add(), ("c", "d"), "output"),
        ],
        "output",
    )
    field = torch.ones(8, dtype=torch.complex64)
    assert torch.allclose(graph(field), (2.0 + 3.0j) * field / 2**0.5)


def test_graph_reports_missing_connections() -> None:
    graph = OpticalChain.from_graph([("node", Scale(1.0), "missing", "out")], "out")
    with pytest.raises(KeyError, match="missing inputs"):
        graph(torch.ones(4))


def test_chain_is_discovered_from_device_port_connections() -> None:
    splitter = PowerSplitter1x2(0.5, trainable=False).named("splitter")
    left = OpticalAttenuator(3.0, trainable=False).named("left")
    right = OpticalAttenuator(6.0, trainable=False).named("right")
    combiner = DirectionalCoupler2x2(0.5, trainable=False).named("combiner")
    splitter.connect_outputs(left, right)
    combiner.connect_inputs(left, right)
    chain = OpticalChain.from_device_graph(splitter, outputs=[(combiner, "output0")])

    field = torch.ones(16, dtype=torch.complex64)
    split0, split1 = splitter(field)
    expected = combiner(left(split0), right(split1))[0]
    assert torch.allclose(chain(field), expected)
    assert set(chain.connected_devices) == {"splitter", "left", "right", "combiner"}


def test_unconnected_optional_output_is_exposed_as_terminal() -> None:
    splitter = PowerSplitter1x2(0.5, trainable=False).named("splitter")
    right = OpticalAttenuator(0.0, trainable=False).named("right")
    splitter.connect_outputs(None, right)
    chain = OpticalChain.from_device_graph(splitter)
    outputs = chain(torch.ones(4, dtype=torch.complex64))
    assert set(outputs) == {"splitter.through", "right.output"}


def test_connected_device_graph_rejects_cycles() -> None:
    first = OpticalAttenuator(0.0).named("first")
    second = OpticalAttenuator(0.0).named("second")
    first.connect(second)
    second.connect(first)
    with pytest.raises(ValueError, match="cycle"):
        OpticalChain.from_device_graph(first)
