import pytest
import torch
from torch import nn

from optosimlab import OpticalChain, PowerSplitter1x2


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
    assert torch.allclose(graph(field), 5.0 * field / 2**0.5)


def test_graph_reports_missing_connections() -> None:
    graph = OpticalChain.from_graph([("node", Scale(1.0), "missing", "out")], "out")
    with pytest.raises(KeyError, match="missing inputs"):
        graph(torch.ones(4))
