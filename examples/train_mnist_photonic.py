"""MNIST：一层可训练光卷积 + 非线性激活 + 电子线性层。"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import struct
from urllib.request import urlopen

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from optosimlab import PhotonicMNISTClassifier


MNIST_BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
MNIST_FILES = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
}


class IDXMNIST(Dataset):
    """不依赖 torchvision 的最小 MNIST IDX 读取器。"""

    def __init__(self, root: Path, *, train: bool) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for filename in MNIST_FILES:
            path = root / filename
            if not path.exists():
                self._download(filename, path)
        prefix = "train" if train else "t10k"
        self.images = self._read_images(root / f"{prefix}-images-idx3-ubyte.gz")
        self.labels = self._read_labels(root / f"{prefix}-labels-idx1-ubyte.gz")

    @staticmethod
    def _download(filename: str, path: Path) -> None:
        errors: list[str] = []
        for base_url in MNIST_BASE_URLS:
            url = base_url + filename
            print(f"downloading {url} ...", flush=True)
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read()
                path.write_bytes(data)
                return
            except OSError as error:
                errors.append(f"{url}: {error}")
        raise RuntimeError("MNIST download failed:\n" + "\n".join(errors))

    @staticmethod
    def _read_images(path: Path) -> Tensor:
        data = gzip.decompress(path.read_bytes())
        magic, count, rows, columns = struct.unpack(">IIII", data[:16])
        if magic != 2051:
            raise ValueError(f"invalid MNIST image file: {path}")
        return torch.frombuffer(bytearray(data[16:]), dtype=torch.uint8).reshape(count, 1, rows, columns).float() / 255.0

    @staticmethod
    def _read_labels(path: Path) -> Tensor:
        data = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", data[:8])
        if magic != 2049:
            raise ValueError(f"invalid MNIST label file: {path}")
        return torch.frombuffer(bytearray(data[8:]), dtype=torch.uint8).reshape(count).long()

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ideal", "nonideal"), default="ideal")
    parser.add_argument("--activation", choices=("relu", "mzi"), default="relu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--chunk-size", type=int, default=0, help="0 表示整段计算")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--synthetic-smoke", action="store_true", help="不用下载数据，只检查训练链路")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.synthetic_smoke:
        generator = torch.Generator().manual_seed(20261018)
        train_images = torch.rand(128, 1, 28, 28, generator=generator)
        train_labels = torch.randint(0, 10, (128,), generator=generator)
        test_images = torch.rand(64, 1, 28, 28, generator=generator)
        test_labels = torch.randint(0, 10, (64,), generator=generator)
        return (
            DataLoader(TensorDataset(train_images, train_labels), batch_size=args.batch_size, shuffle=True),
            DataLoader(TensorDataset(test_images, test_labels), batch_size=args.batch_size),
        )

    train_data = IDXMNIST(args.data_dir, train=True)
    test_data = IDXMNIST(args.data_dir, train=False)
    if args.limit_train:
        train_data = Subset(train_data, range(min(args.limit_train, len(train_data))))
    if args.limit_test:
        test_data = Subset(test_data, range(min(args.limit_test, len(test_data))))
    return (
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(test_data, batch_size=args.batch_size, num_workers=0),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            prediction = model(images.to(device)).argmax(dim=1)
            correct += int((prediction == labels.to(device)).sum())
            count += labels.numel()
    return correct / max(count, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(20261018)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("请求了 CUDA，但当前 PyTorch/CUDA 不可用")

    train_loader, test_loader = make_loaders(args)
    model = PhotonicMNISTClassifier(
        mode=args.mode,
        activation=args.activation,
        chunk_size=args.chunk_size or None,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    print(json.dumps(model.nonideality_description, ensure_ascii=False, indent=2))
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            model.project_physical_parameters_()
            running_loss += float(loss.detach()) * labels.numel()
            sample_count += labels.numel()
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch:02d} loss={running_loss / sample_count:.6f} test_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
