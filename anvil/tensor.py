"""Anvil's core: an n-dimensional Tensor with reverse-mode autograd (NumPy backend)."""
 
import numpy as np
 
 
def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`: the reverse of NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad=grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s==1 and grad.shape[i] != 1:
            grad=grad.sum(axis=i, keepdims=True)
    return grad
 
 
def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)
 
 
def _result(data, parents):
    """Build an op's output. Only record the graph if some input needs a gradient."""
    needs_grad=any(p.requires_grad for p in parents)
    return Tensor(data, requires_grad=needs_grad, _parents=parents if needs_grad else ())
 
 
class Tensor:
    __array_ufunc__=None  # make `numpy_array * Tensor` defer to Tensor's operators
 
    def __init__(self, data, requires_grad=False, _parents=()):
        if isinstance(data, Tensor):
            data=data.data
        if isinstance(data, (np.ndarray, np.generic)):
            arr=np.asarray(data)
            if not np.issubdtype(arr.dtype, np.floating):
                arr=arr.astype(np.float32)
        else:  # python numbers / lists default to float32, like PyTorch
            arr=np.asarray(data, dtype=np.float32)
        self.data=arr
        self.grad=None
        self.requires_grad=requires_grad
        self._parents=_parents
        self._backward=lambda: None
 
    #basic properties
    @property
    def shape(self):
        return self.data.shape
 
    @property
    def ndim(self):
        return self.data.ndim
 
    @property
    def size(self):
        return self.data.size
 
    @property
    def T(self):
        return self.transpose()
 
    def item(self):
        return self.data.item()
 
    def numpy(self):
        return self.data
 
    def detach(self):
        return Tensor(self.data)
 
    def zero_grad(self):
        self.grad=None
 
    def __repr__(self):
        flag=", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{flag})"
 
    # ---------- autograd machinery ----------
    def _accum(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad=np.array(g)  # copy, so we never alias another tensor's grad
        else:
            self.grad=self.grad + g
 
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() needs a scalar output, or pass `grad` explicitly")
            grad=np.ones_like(self.data)
 
        # iterative topological sort (no recursion-limit problems on deep graphs)
        topo, visited, stack=[], set(), [(self, False)]
        while stack:
            node, processed=stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
 
        # clear stale grads on intermediate nodes so calling backward twice is safe;
        # leaf tensors (parameters) keep accumulating, like PyTorch
        for t in topo:
            if t._parents:
                t.grad=None
        self.grad=np.array(grad)
 
        for t in reversed(topo):
            t._backward()
 
    #elementwise arithmetic
    def __add__(self, other):
        other=_as_tensor(other)
        out=_result(self.data + other.data, (self, other))
 
        def _backward():
            self._accum(_unbroadcast(out.grad, self.data.shape))
            other._accum(_unbroadcast(out.grad, other.data.shape))
 
        out._backward=_backward
        return out
 
    __radd__=__add__
 
    def __mul__(self, other):
        other=_as_tensor(other)
        out=_result(self.data * other.data, (self, other))
 
        def _backward():
            self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.data.shape))
 
        out._backward=_backward
        return out
 
    __rmul__=__mul__
 
    def __neg__(self):
        out=_result(-self.data, (self,))
 
        def _backward():
            self._accum(-out.grad)
 
        out._backward=_backward
        return out
 
    def __sub__(self, other):
        return self + (-_as_tensor(other))
 
    def __rsub__(self, other):
        return _as_tensor(other) + (-self)
 
    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only int/float exponents are supported"
        out=_result(self.data ** p, (self,))
 
        def _backward():
            self._accum(out.grad * p * self.data ** (p - 1))
 
        out._backward=_backward
        return out
 
    def __truediv__(self, other):
        return self * (_as_tensor(other) ** -1)
 
    def __rtruediv__(self, other):
        return _as_tensor(other) * (self ** -1)
 
    #elementwise math
    def exp(self):
        out=_result(np.exp(self.data), (self,))
 
        def _backward():
            self._accum(out.grad * out.data)
 
        out._backward=_backward
        return out
 
    def log(self):
        out=_result(np.log(self.data), (self,))
 
        def _backward():
            self._accum(out.grad / self.data)
 
        out._backward=_backward
        return out
 
    def relu(self):
        out=_result(np.maximum(self.data, 0), (self,))
 
        def _backward():
            self._accum(out.grad * (self.data > 0))
 
        out._backward=_backward
        return out
 
    def tanh(self):
        out=_result(np.tanh(self.data), (self,))
 
        def _backward():
            self._accum(out.grad * (1 - out.data ** 2))
 
        out._backward=_backward
        return out
 
    def sigmoid(self):
        out=_result(1.0 / (1.0 + np.exp(-self.data)), (self,))
 
        def _backward():
            self._accum(out.grad * out.data * (1 - out.data))
 
        out._backward=_backward
        return out
 
    #reductions
    def sum(self, axis=None, keepdims=False):
        out=_result(self.data.sum(axis=axis, keepdims=keepdims), (self,))
 
        def _backward():
            g=out.grad
            if axis is not None and not keepdims:
                g=np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.data.shape))
 
        out._backward=_backward
        return out
 
    def mean(self, axis=None, keepdims=False):
        s=self.sum(axis=axis, keepdims=keepdims)
        n=self.data.size // s.data.size
        return s * (1.0 / n)
 
    def max(self, axis=None, keepdims=False):
        out=_result(self.data.max(axis=axis, keepdims=keepdims), (self,))
 
        def _backward():
            m=self.data.max(axis=axis, keepdims=True)
            mask=(self.data==m).astype(self.data.dtype)
            mask=mask / mask.sum(axis=axis, keepdims=True)  # split ties evenly
            g=out.grad
            if axis is not None and not keepdims:
                g=np.expand_dims(g, axis)
            self._accum(mask * g)
 
        out._backward=_backward
        return out
 
    #linear algebra
    def matmul(self, other):
        """Matrix multiply for tensors with ndim >= 2 (batch dims broadcast)."""
        other=_as_tensor(other)
        assert self.ndim >= 2 and other.ndim >= 2, "matmul needs ndim >= 2 (reshape vectors first)"
        out=_result(self.data @ other.data, (self, other))
 
        def _backward():
            a, b=self.data, other.data
            ga=out.grad @ np.swapaxes(b, -1, -2)
            gb=np.swapaxes(a, -1, -2) @ out.grad
            self._accum(_unbroadcast(ga, a.shape))
            other._accum(_unbroadcast(gb, b.shape))
 
        out._backward=_backward
        return out
 
    __matmul__=matmul
 
    def __rmatmul__(self, other):
        return _as_tensor(other).matmul(self)
 
    #shape ops
    def reshape(self, *shape):
        if len(shape)==1 and isinstance(shape[0], (tuple, list)):
            shape=tuple(shape[0])
        out=_result(self.data.reshape(shape), (self,))
 
        def _backward():
            self._accum(out.grad.reshape(self.data.shape))
 
        out._backward=_backward
        return out
 
    def transpose(self, *axes):
        if len(axes)==1 and isinstance(axes[0], (tuple, list)):
            axes=tuple(axes[0])
        perm=tuple(a % self.ndim for a in axes) if axes else tuple(reversed(range(self.ndim)))
        inverse=tuple(np.argsort(perm))
        out=_result(self.data.transpose(perm), (self,))
 
        def _backward():
            self._accum(out.grad.transpose(inverse))
 
        out._backward=_backward
        return out
 
    def __getitem__(self, idx):
        out=_result(self.data[idx], (self,))
 
        def _backward():
            g=np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)  # add.at handles repeated indices correctly
            self._accum(g)
 
        out._backward=_backward
        return out
 

