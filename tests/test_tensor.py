import numpy as np
import pytest

from anvil.tensor import Tensor



#Numerical gradient checking helpers

def numeric_grads(loss_fn, arrays, eps=1e-6):
    """Central finite differences: (f(x+eps) - f(x-eps)) / 2eps, one element at a time."""
    grads=[]
    for a in arrays:
        g=np.zeros_like(a)
        for idx in np.ndindex(*a.shape):
            orig=a[idx]
            a[idx]=orig + eps
            f_plus=loss_fn(*[Tensor(x) for x in arrays]).item()
            a[idx]=orig - eps
            f_minus=loss_fn(*[Tensor(x) for x in arrays]).item()
            a[idx]=orig
            g[idx]=(f_plus - f_minus) / (2 * eps)
        grads.append(g)
    return grads


def check(fn, *shapes, positive=False, atol=1e-5):
    """Compare Anvil's analytic gradients of fn against numerical ones."""
    rng=np.random.default_rng(0)
    make=(lambda s: rng.uniform(0.5, 2.0, s)) if positive else (lambda s: rng.standard_normal(s))
    arrays=[make(s) for s in shapes]  #float64 so the comparison is tight

    probe=fn(*[Tensor(a) for a in arrays])
    weights=rng.standard_normal(probe.shape)  #random projection -> scalar loss

    def loss_fn(*ts):
        return (fn(*ts) * Tensor(weights)).sum()

    tensors=[Tensor(a.copy(), requires_grad=True) for a in arrays]
    loss_fn(*tensors).backward()

    for t, numeric in zip(tensors, numeric_grads(loss_fn, arrays)):
        assert t.grad is not None
        assert t.grad.shape==t.data.shape
        np.testing.assert_allclose(t.grad, numeric, atol=atol)



#The very first test: keep it

def test_mul_grad():
    a=Tensor([2.0], requires_grad=True)
    b=Tensor([3.0], requires_grad=True)
    c=a * b
    c.backward()
    assert a.grad[0]==3.0
    assert b.grad[0]==2.0



#Elementwise ops (including broadcasting)

@pytest.mark.parametrize("shapes", [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),      #row broadcast
    ((3, 1), (1, 4)),    #both inputs broadcast
    ((2, 3, 4), (4,)),   #extra leading dims
    ((3, 4), ()),        #scalar tensor
])
@pytest.mark.parametrize("op", [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
], ids=["add", "sub", "mul"])
def test_binary_broadcasting(op, shapes):
    check(op, *shapes)


@pytest.mark.parametrize("shapes", [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4))])
def test_div(shapes):
    check(lambda a, b: a / b, *shapes, positive=True)


def test_neg():
    check(lambda a: -a, (3, 4))


@pytest.mark.parametrize("p", [2, 3, -1, 0.5])
def test_pow(p):
    check(lambda a: a ** p, (3, 4), positive=True)


def test_reverse_operators_with_python_numbers():
    check(lambda a: 2.0 + a, (3, 4))
    check(lambda a: 2.0 - a, (3, 4))
    check(lambda a: 2.0 * a, (3, 4))
    check(lambda a: 2.0 / a, (3, 4), positive=True)


@pytest.mark.parametrize("fn", [
    lambda a: a.exp(),
    lambda a: a.relu(),
    lambda a: a.tanh(),
    lambda a: a.sigmoid(),
], ids=["exp", "relu", "tanh", "sigmoid"])
def test_unary(fn):
    check(fn, (3, 4))


def test_log():
    check(lambda a: a.log(), (3, 4), positive=True)



#Reductions

@pytest.mark.parametrize("axis,keepdims", [
    (None, False), (0, False), (1, False), (1, True), (-1, False), ((0, 2), False),
])
def test_sum(axis, keepdims):
    check(lambda a: a.sum(axis=axis, keepdims=keepdims), (2, 3, 4))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), (-1, False)])
def test_mean(axis, keepdims):
    check(lambda a: a.mean(axis=axis, keepdims=keepdims), (2, 3, 4))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), (-1, False)])
def test_max(axis, keepdims):
    check(lambda a: a.max(axis=axis, keepdims=keepdims), (2, 3, 4))


def test_max_splits_gradient_on_ties():
    x=Tensor([1.0, 3.0, 3.0], requires_grad=True)
    x.max().backward()
    np.testing.assert_allclose(x.grad, [0.0, 0.5, 0.5])



#Matmul and shape ops

def test_matmul_2d():
    check(lambda a, b: a @ b, (3, 4), (4, 5))


def test_matmul_batched():
    check(lambda a, b: a @ b, (2, 3, 4), (2, 4, 5))


def test_matmul_batch_broadcast():
    check(lambda a, b: a @ b, (2, 3, 4), (4, 5))


def test_reshape():
    check(lambda a: a.reshape(6, 2), (3, 4))
    check(lambda a: a.reshape((2, -1)), (3, 4))


def test_transpose():
    check(lambda a: a.T, (3, 4))
    check(lambda a: a.transpose(1, 0, 2), (2, 3, 4))
    check(lambda a: a.transpose(2, 0, 1), (2, 3, 4))


def test_getitem():
    check(lambda a: a[1:3], (4, 5))
    check(lambda a: a[:, 2], (4, 5))
    check(lambda a: a[[0, 2, 2]], (4, 5))  #repeated index: grads must add up



#Graph behaviour

def test_tensor_used_twice():
    #y=x*x + x  ->  dy/dx=2x + 1
    x=Tensor([3.0], requires_grad=True)
    y=x * x + x
    y.backward()
    assert x.grad[0]==pytest.approx(7.0)


def test_composite_expression():
    check(lambda a, b: ((a @ b).tanh() * 2.0 + 1.0).relu().mean(axis=0), (4, 3), (3, 5))


def test_backward_twice_is_stable_for_intermediates():
    x=Tensor([2.0], requires_grad=True)
    y=(x * x) * 3.0          #dy/dx=6x=12
    y.backward()
    y.backward()               #leaves accumulate (like PyTorch), intermediates must not
    assert x.grad[0]==pytest.approx(24.0)


def test_no_grad_tracking_without_requires_grad():
    a, b=Tensor([1.0, 2.0]), Tensor([3.0, 4.0])
    c=(a * b).sum()
    assert not c.requires_grad
    assert c._parents==()


def test_backward_requires_scalar():
    x=Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_deep_graph_does_not_hit_recursion_limit():
    x=Tensor([1.0], requires_grad=True)
    y=x
    for _ in range(5000):
        y=y + 1.0
    y.backward()
    assert x.grad[0]==pytest.approx(1.0)



#Comparison with PyTorch (skipped automatically if torch isn't installed)

def test_matches_pytorch():
    torch=pytest.importorskip("torch")
    rng=np.random.default_rng(1)
    x_np=rng.standard_normal((5, 3))
    w_np=rng.standard_normal((3, 4))
    b_np=rng.standard_normal((4,))

    x, w, b=(Tensor(v, requires_grad=True) for v in (x_np, w_np, b_np))
    ((x @ w + b).tanh() ** 2).mean().backward()

    tx, tw, tb=(torch.tensor(v, requires_grad=True) for v in (x_np, w_np, b_np))
    ((tx @ tw + tb).tanh() ** 2).mean().backward()

    np.testing.assert_allclose(x.grad, tx.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(w.grad, tw.grad.numpy(), atol=1e-6)
    np.testing.assert_allclose(b.grad, tb.grad.numpy(), atol=1e-6)