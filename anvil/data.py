"""Data utilities: DataLoader and an MNIST loader (downloads once, then caches)."""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .rng import get_rng
from .tensor import Tensor


class DataLoader:
    """Iterate over (x_batch Tensor, y_batch ndarray) pairs from in-memory arrays."""

    def __init__(self, x, y, batch_size=64, shuffle=False, drop_last=False, seed=None):
        assert len(x)==len(y), "x and y must have the same length"
        self.x, self.y=np.asarray(x), np.asarray(y)
        self.batch_size, self.shuffle, self.drop_last=batch_size, shuffle, drop_last
        self.rng=np.random.default_rng(seed) if seed is not None else None

    def __len__(self):
        n=len(self.x)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        idx=np.arange(len(self.x))
        if self.shuffle:
            (self.rng or get_rng()).shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch=idx[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield Tensor(self.x[batch]), self.y[batch]


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------
_MIRRORS=[
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]
_FILES={
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _parse_idx_images(raw):
    magic, n, rows, cols=struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"bad image file (magic number {magic})")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def _parse_idx_labels(raw):
    magic, n=struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"bad label file (magic number {magic})")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(n)


def _download(filename, dest):
    errors=[]
    for base in _MIRRORS:
        try:
            req=urllib.request.Request(base + filename, headers={"User-Agent": "anvil"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as e:  # try the next mirror
            errors.append(f"{base}: {e}")
    raise RuntimeError(
        "Could not download MNIST. Download the four .gz files manually into "
        f"{dest.parent} and run again.\n" + "\n".join(errors)
    )


def load_mnist(root=None):
    """Return ((x_train, y_train), (x_test, y_test)).

    Images are float32 in [0, 1] with shape (N, 28, 28); labels are int64 (N,).
    Files are cached in ~/.anvil/mnist (or `root`).
    """
    root=Path(root) if root else Path.home() / ".anvil" / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    arrays={}
    for key, filename in _FILES.items():
        path=root / filename
        if not path.exists():
            print(f"Downloading {filename} ...")
            _download(filename, path)
        raw=gzip.decompress(path.read_bytes())
        arrays[key]=_parse_idx_images(raw) if "images" in key else _parse_idx_labels(raw)
    x_train=arrays["train_images"].astype(np.float32) / 255.0
    x_test=arrays["test_images"].astype(np.float32) / 255.0
    return (x_train, arrays["train_labels"].astype(np.int64)), (x_test, arrays["test_labels"].astype(np.int64))