"""Stateless functions built from Tensor ops (so gradients come for free)"""

import numpy as np

from ..tensor import Tensor

def log_softmax(x, axis=-1):
    """Numerically stable: subtract the max first s exp() can't overflow"""
    shifted=x-Tensor(x.data.max(axiis=axis, keepdims=True))
    return shifted-shifted.exp().sum(axis=axis, keepdims=True).log()


def softmax(x, axis=-1):
    return log_softmax(x, axis=axis).exp()


def cross_entropy(logits, targets):
    """Mean cross-entropy between raw logits (N, C) and integer class labels (N, )"""
    targets=np.asarray(targets)
    if logits.ndim !=2 or targets.shape !=(logits.shape[0],):
        raise ValueError(f"expected logits (N, C) and targets (N, ), got {logits.shape} and {targets.shape}")

    targets=targets.astype(np.int64)
    if targets.min()<0 or targets.max() >=logits.shape[1]:
        raise ValueError("target class index out of range")
    logp=log_softmax(logits, axis=-1)
    picked=logp[np.arange(logits.shape[0]), targets]
    return -picked.mean()



def mse_loss(pred, target):
    diff=pred-target
    return (diff*diff).mean()