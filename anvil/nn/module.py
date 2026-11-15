"""Module and Parameter: the building blocks every layer inherits from."""

import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """A Tensor that a Module treats as a learnable weight (always requires grad)."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


def _walk(value, prefix):
    """Yield (name, item) for every Parameter/Module inside value (also inside lists/dicts)."""
    if isinstance(value, (Parameter, Module)):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk(v, f"{prefix}.{i}")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{prefix}.{k}")


class Module:
    training=True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # finding parameters and submodules
    def _collect(self, out, seen, prefix):
        for name, value in self.__dict__.items():
            for n, item in _walk(value, name):
                if isinstance(item, Parameter):
                    if id(item) not in seen:
                        seen.add(id(item))
                        out.append((prefix + n, item))
                else:
                    item._collect(out, seen, prefix + n + ".")

    def named_parameters(self):
        out=[]
        self._collect(out, set(), "")
        return out

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def modules(self):
        """This module and every module nested inside it."""
        out, seen=[], set()

        def rec(m):
            if id(m) in seen:
                return
            seen.add(id(m))
            out.append(m)
            for value in m.__dict__.values():
                for _, item in _walk(value, ""):
                    if isinstance(item, Module):
                        rec(item)

        rec(self)
        return out

    # ---------- modes and gradients ----------
    def train(self, mode=True):
        for m in self.modules():
            m.training=mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad=None

    # ---------- saving and loading ----------
    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        params=dict(self.named_parameters())
        if set(params) != set(state):
            raise KeyError(f"state_dict keys don't match: missing={set(params) - set(state)}, "
                           f"unexpected={set(state) - set(params)}")
        for name, p in params.items():
            if p.data.shape != np.shape(state[name]):
                raise ValueError(f"shape mismatch for {name}: {p.data.shape} vs {np.shape(state[name])}")
            p.data[...]=state[name]

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as f:
            self.load_state_dict({k: f[k] for k in f.files})

    def extra_repr(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.extra_repr()})"