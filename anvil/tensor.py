import numpy as np

def unbroadcast(grad, shape):
    """Sum a gradient back down to the shape of the original input."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = lambda: None

    def _accum(self, g):
        self.grad = g if self.grad is None else self.grad + g

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other))
        def _backward():
            if self.requires_grad:
                self._accum(unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other))
        def _backward():
            if self.requires_grad:
                self._accum(unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accum(unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other))
        def _backward():
            if self.requires_grad:
                self._accum(out.grad @ other.data.T)
            if other.requires_grad:
                other._accum(self.data.T @ out.grad)
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    build(p)
                topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()