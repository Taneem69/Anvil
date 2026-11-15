"""Layers: Linear, activations, Flatten, Dropout, Sequential."""

import numpy as np

from ..rng import get_rng
from ..tensor import Tensor
from .module import Module, Parameter


class Linear(Module):
    """y=x @ W + b, with PyTorch's default init: uniform(-1/sqrt(in), 1/sqrt(in))."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features, self.out_features=in_features, out_features
        bound=1.0 / np.sqrt(in_features)
        rng=get_rng()
        self.weight=Parameter(rng.uniform(-bound, bound, (in_features, out_features)).astype(np.float32))
        self.bias=Parameter(rng.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None

    def forward(self, x):
        out=x @ self.weight
        return out + self.bias if self.bias is not None else out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Flatten(Module):
    """(N, ...) -> (N, features)."""

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Dropout(Module):
    """Randomly zeroes a fraction p of values while training; identity in eval mode."""

    def __init__(self, p=0.5):
        assert 0.0 <= p < 1.0, "p must be in [0, 1)"
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep=(get_rng().random(x.shape) >= self.p).astype(x.data.dtype) / (1.0 - self.p)
        return x * Tensor(keep)

    def extra_repr(self):
        return f"p={self.p}"


class Sequential(Module):
    def __init__(self, *layers):
        self.layers=list(layers)

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        inner="\n".join(f"  ({i}): {layer!r}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{inner}\n)"