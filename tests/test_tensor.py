from anvil.tensor import Tensor

def test_mul_grad():
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a * b
    c.backward()
    assert a.grad[0] == 3.0
    assert b.grad[0] == 2.0