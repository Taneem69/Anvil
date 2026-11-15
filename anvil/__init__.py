"""Anvil: a deep learning framework built fro scratch"""

from .tensor import Tensor, no_grad
from .rng import manual_seed
from . import nn, optim, data


__version__="0.1.0"