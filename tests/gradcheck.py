"""Shared helper: sompare analytic gradients with finits-difference estimates"""

import numpy as np
from anvil.tensor import Tensor

def numeric_grads(loss_fn, arrays, eps=1e-6):
    grads=[]
    for a in arrays:
        g=np.zeros_like(a)
        for idx in np.ndindex(*a.shape):
            orig=a[idx]
            a[idx]=orig+eps
            f_plus=loss_fn(*[Tensor(x) for x in arrays]).item()
            a[idx]=orig-eps
            f_minus=loss_fn(*[Tensor(x) for x in arrays]).item()
            a[idx]=orig
            g[idx]=(f_plus-f_minus)/(2*eps)

        grads.append(g)

    return grads




def check(fn, *shapes, positive=False, atol=1e-5):
    rng=np.random.default_rng(0)
    make=(lambda s: rng.uniform(0.5, 2.0, s)) if positive else (lambda s: rng.standard_normal(s))


    arrays=[make(s) for s in shapes]
    probe=fn(*[Tensor(a) for a in arrays])
    weights=rng.standard_normal(probe.shape)


    def loss_fn(*ts):
        return (fn(*ts) * Tensor(weights)).sum()


    tensors=[Tensor(a.copy(), requires_grad=True) for a in arrays]
    loss_fn(*tensors).backward()

    for t, numeric in zip(tensors, numeric_grads(loss_fn, arrays)):
        assert t.grad is not None and t.grad.shape==t.data.shape
        np.testing.assert_allclose(t.grad, numeric, atol=atol)

