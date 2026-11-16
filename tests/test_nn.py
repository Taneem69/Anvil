import gzip
import os
import struct
import tempfile

import numpy as np
import pytest


import anvil
from anvil import nn, optim
from anvil.data import DataLoader, load_mnist, _parse_idx_images, _parse_idx_labels
from anvil.tensor import Tensor, no_grad
from gradcheck import check


F=nn.functional


#Functional: log_softmax/cross_entropy/mse

def test_log_softmax_grad():
    check(lambda a :F.log_softmax(a, axis=-1), (4,5))



def test_softmax_rows_sum_to_one():
    p=F.softmax(Tensor(np.random.default_rng(0).standard_normal((6,7))))
    np.testing.assert_allclose(p.data.sum(axis=-1), np.ones(6), atol=1e-6)



def test_log_softmax_is_stable_for_huge_logits():
    out=F.log_softmax(Tensor([[1000.0,1001.0,999.0]]))
    assert np.all(np.isfinite(out.data))
    np.testing.assert_allclose(np.exp(out.data).sum(), 1.0, atol=1e-5)


def test_cross_entropy_uniform_logits_is_log_c():
    loss=F.cross_entropy(Tensor(np.zeros((8,10))), np.arange(8))
    assert loss.item()==pytest.approx(np.log(10), rel=1e-5)


def test_cross_entropy_grad_matches_softmax_minus_oneshot():
    rng=np.random.default_rng(1)
    logits_np=rng.standard_normal((5,4))
    targets=np.array([0,3,1,1,2])
    logits=Tensor(logits_np, requires_grad=True)
    F.cross_entropy(logits, targets).backward()
    probs=np.exp(logits_np-logits_np.max(axis=1, keepdims=True))
    probs/=probs.sum(axis=1, keepdims=True)
    probs[np.arange(5), targets] -=1.0
    np.testing.assert_allclose(logits.grad, probs/5, atol=1e-9)



def test_cross_entropy_numeric_grad():
    targets=np.array([2,0,1])
    check(lambda a: F.cross_entropy(a, targets).reshape(1), (3,4))


def test_cross_entropy_rejects_bad_targets():
    with pytest.raises(ValueError):
        F.cross_entropy(Tensor(np.zeros((2,3))), np.array([0,3]))

    with pytest.raises(ValueError):
        F.cross_entropy(Tensor(np.zeros((2,3))), np.array([0,1,2]))



def test_mse_loss():
    check(lambda a, b: F.mse_loss(a,b).reshape(1), (3,4), (3,4))
    assert F.mse_loss(Tensor([1.0,3.0]), Tensor([0.0, 0.0])).item()==pytest.approx(5.0)




#Moduule Mechanics

def make_model():
    anvil.manual_seed(0)
    return nn.Sequential(nn.Linear(3,5), nn.Tanh(), nn.Dropout(0.3), nn.Linear(5,4))


def test_parameters_are_found_in_nested_modules():
    model=make_model()
    names=[n for n, _ in model.named_parameters()]
    assert names==["layers.0.weight", "layers.0.bias", "layers.3.weight", "layers.3.bias"]

    assert model.num_parameters()==3*5+5+5*4+4




def test_parameters_found_inside_lists_and_shared_params_counted_once():
    class Net(nn.Module):
        def __init__(self):
            self.blocks=[nn.Linear(2, 2), nn.Linear(2, 2)]
            self.shared=self.blocks[0]
    assert len(Net().parameters())==4


def test_linear_without_bias():
    layer=nn.Linear(3, 2, bias=False)
    assert len(layer.parameters())==1
    assert layer(Tensor(np.ones((4, 3)))).shape==(4, 2)


def test_train_eval_propagates_to_children():
    model=make_model()
    model.eval()
    assert all(not m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())


def test_zero_grad_clears_gradients():
    model=make_model()
    model.eval()
    F.cross_entropy(model(Tensor(np.ones((2, 3)))), np.array([0, 1])).backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_state_dict_roundtrip_and_save_load():
    a, b=make_model(), make_model()
    for p in b.parameters():
        p.data[...]=0.0
    b.load_state_dict(a.state_dict())
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(pa.data, pb.data)

    with tempfile.TemporaryDirectory() as d:
        path=os.path.join(d, "w.npz")
        a.save(path)
        c=make_model()
        for p in c.parameters():
            p.data[...]=1.0
        c.load(path)
    for pa, pc in zip(a.parameters(), c.parameters()):
        np.testing.assert_array_equal(pa.data, pc.data)


def test_load_state_dict_rejects_mismatch():
    model=make_model()
    with pytest.raises(KeyError):
        model.load_state_dict({"nope": np.zeros(1)})
    bad=model.state_dict()
    bad["layers.0.weight"]=np.zeros((9, 9))
    with pytest.raises(ValueError):
        model.load_state_dict(bad)


def test_full_model_parameter_gradients_match_numeric():
    anvil.manual_seed(3)
    model=nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4))
    params=model.parameters()
    for p in params:  
        p.data=p.data.astype(np.float64)
    x=Tensor(np.random.default_rng(0).standard_normal((6, 3)))
    targets=np.array([0, 1, 2, 3, 0, 1])

    def loss_fn():
        return F.cross_entropy(model(x), targets)

    model.zero_grad()
    loss_fn().backward()
    for p in params:
        numeric=np.zeros_like(p.data)
        for idx in np.ndindex(*p.data.shape):
            orig=p.data[idx]
            p.data[idx]=orig + 1e-6
            plus=loss_fn().item()
            p.data[idx]=orig - 1e-6
            minus=loss_fn().item()
            p.data[idx]=orig
            numeric[idx]=(plus - minus) / 2e-6
        np.testing.assert_allclose(p.grad, numeric, atol=1e-6)


def test_flatten():
    assert nn.Flatten()(Tensor(np.zeros((5, 2, 3, 4)))).shape==(5, 24)


def test_dropout_eval_is_identity_and_train_preserves_mean():
    anvil.manual_seed(0)
    x=Tensor(np.ones((200, 200)))
    layer=nn.Dropout(0.4)
    layer.eval()
    assert layer(x) is x
    layer.train()
    out=layer(x).data
    assert abs((out==0).mean() - 0.4) < 0.02
    assert abs(out.mean() - 1.0) < 0.05  


def test_no_grad_builds_no_graph():
    p=nn.Parameter(np.ones((2, 2), dtype=np.float32))
    with no_grad():
        out=(p * 2.0).sum()
    assert not out.requires_grad and out._parents==()
    assert (p * 2.0).sum().requires_grad 


# Optimizers (checked against a plain NumPy reference of the textbook formulas)
def _grads(seed, steps, shape):
    rng=np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(steps)]


def test_sgd_plain_converges_on_quadratic():
    p=nn.Parameter(np.array([5.0, -3.0], dtype=np.float32))
    opt=optim.SGD([p], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        (p * p).sum().backward()
        opt.step()
    np.testing.assert_allclose(p.data, 0.0, atol=1e-4)


def test_sgd_momentum_and_weight_decay_match_reference():
    grads=_grads(0, 6, (4,))
    w0=np.random.default_rng(1).standard_normal(4).astype(np.float32)
    p=nn.Parameter(w0.copy())
    opt=optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)
    ref, v=w0.astype(np.float64), np.zeros(4)
    for g in grads:
        p.grad=g.copy()
        opt.step()
        gg=g + 0.01 * ref
        v=0.9 * v + gg
        ref=ref - 0.1 * v
    np.testing.assert_allclose(p.data, ref, atol=1e-5)


def test_adam_matches_reference():
    grads=_grads(2, 8, (3, 2))
    w0=np.random.default_rng(3).standard_normal((3, 2)).astype(np.float32)
    p=nn.Parameter(w0.copy())
    opt=optim.Adam([p], lr=0.01, weight_decay=0.1)
    ref, m, v=w0.astype(np.float64), np.zeros((3, 2)), np.zeros((3, 2))
    for t, g in enumerate(grads, start=1):
        p.grad=g.copy()
        opt.step()
        gg=g + 0.1 * ref
        m=0.9 * m + 0.1 * gg
        v=0.999 * v + 0.001 * gg * gg
        ref=ref - (0.01 / (1 - 0.9 ** t)) * m / (np.sqrt(v) / np.sqrt(1 - 0.999 ** t) + 1e-8)
    np.testing.assert_allclose(p.data, ref, atol=1e-5)


def test_optimizers_skip_params_without_grad():
    p=nn.Parameter(np.ones(3, dtype=np.float32))
    optim.SGD([p], lr=1.0).step()
    optim.Adam([p], lr=1.0).step()
    np.testing.assert_array_equal(p.data, np.ones(3))


def test_optimizers_match_pytorch():
    torch=pytest.importorskip("torch")
    grads=_grads(4, 6, (5,))
    w0=np.random.default_rng(5).standard_normal(5).astype(np.float32)
    configs=[
        (lambda p: optim.SGD([p], lr=0.05, momentum=0.9, weight_decay=0.01),
         lambda p: torch.optim.SGD([p], lr=0.05, momentum=0.9, weight_decay=0.01)),
        (lambda p: optim.Adam([p], lr=0.01, weight_decay=0.01),
         lambda p: torch.optim.Adam([p], lr=0.01, weight_decay=0.01)),
    ]
    for make_ours, make_torch in configs:
        p=nn.Parameter(w0.copy())
        tp=torch.tensor(w0.copy(), requires_grad=True)
        ours, theirs=make_ours(p), make_torch(tp)
        for g in grads:
            p.grad=g.copy()
            tp.grad=torch.tensor(g.copy())
            ours.step()
            theirs.step()
        np.testing.assert_allclose(p.data, tp.detach().numpy(), rtol=1e-4, atol=1e-5)



#Data
def test_dataloader_covers_every_sample_once():
    x, y=np.arange(23, dtype=np.float32).reshape(23,1), np.arange(23)
    loader=DataLoader(x, y, batch_size=5, shuffle=True, seed=0)
    assert len(loader)==5
    seen=np.concatenate([yb for _, yb in loader])
    assert sorted(seen.tolist())==list(range(23))



def test_dataloader_drop_last_and_seed_reproducibility():
    x, y=np.zeros((23, 2), dtype=np.float32), np.arange(23)
    assert len(DataLoader(x, y, batch_size=5, drop_last=True))==4
    assert all(len(yb)==5 for _, yb in DataLoader(x, y, batch_size=5, drop_last=True))
    first=[yb.tolist() for _, yb in DataLoader(x, y, batch_size=5, shuffle=True, seed=7)]
    second=[yb.tolist() for _, yb in DataLoader(x, y, batch_size=5, shuffle=True, seed=7)]
    assert first==second


def test_dataloader_yields_tensors():
    xb, yb=next(iter(DataLoader(np.zeros((4, 2), dtype=np.float32), np.arange(4), batch_size=2)))
    assert isinstance(xb, Tensor) and xb.shape==(2, 2) and yb.shape==(2,)


def test_mnist_idx_parsing_and_cached_loading():
    rng=np.random.default_rng(0)
    imgs=lambda n: rng.integers(0, 256, (n, 28, 28), dtype=np.uint8)
    lbls=lambda n: rng.integers(0, 10, n, dtype=np.uint8)
    tr_x, tr_y, te_x, te_y=imgs(12), lbls(12), imgs(5), lbls(5)

    def image_file(a):
        return struct.pack(">IIII", 2051, *a.shape) + a.tobytes()

    def label_file(a):
        return struct.pack(">II", 2049, len(a)) + a.tobytes()

    with tempfile.TemporaryDirectory() as d:
        files={
            "train-images-idx3-ubyte.gz": image_file(tr_x), "train-labels-idx1-ubyte.gz": label_file(tr_y),
            "t10k-images-idx3-ubyte.gz": image_file(te_x), "t10k-labels-idx1-ubyte.gz": label_file(te_y),
        }
        for name, raw in files.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(gzip.compress(raw))
        (xt, yt), (xe, ye)=load_mnist(root=d) 
    np.testing.assert_allclose(xt, tr_x / 255.0, atol=1e-6)
    np.testing.assert_array_equal(yt, tr_y)
    assert xt.dtype==np.float32 and yt.dtype==np.int64 and xe.shape==(5, 28, 28)
    np.testing.assert_array_equal(ye, te_y)


def test_idx_parsers_reject_wrong_files():
    with pytest.raises(ValueError):
        _parse_idx_images(struct.pack(">IIII", 1234, 1, 28, 28))
    with pytest.raises(ValueError):
        _parse_idx_labels(struct.pack(">II", 1234, 1))



#End to end: the framework actually learns

def test_mlp_learns_three_class_spiral():
    anvil.manual_seed(0)
    rng=np.random.default_rng(0)
    n, k=100, 3
    x=np.zeros((n*k, 2), dtype=np.float32)
    y=np.zeros(n*k, dtype=np.int64)
    for c in range(k):
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.standard_normal(n) * 0.2
        x[c * n:(c + 1) * n] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[c * n:(c + 1) * n] = c

    model=nn.Sequential(nn.Linear(2,64), nn.Tanh(), nn.linear(64, 64), nn.Tanh(), nn.Linear(64, k))
    opt=optim.Adam(model.parameters(), lr=0.02)
    first_loss=None
    for step in range(400):
        loss=F.cross_entropy(model(Tensor(x)), y)
        first_loss=first_loss if first_loss is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc=(model(Tensor(x)).data.argmax(axis=1)==y).mean()
    assert loss.item()<first_loss*0.3
    assert acc>0.95


#Regression: constant sub-expressions inside a graph much not break backward()

def test_contant_subexpressions_dont_break_backward():
    x=Tensor([1.0, 2.0, 3.0], requires_grad=True)
    c=Tensor([4.0, 5.0, 6.0])
    y=(x-c(-c)*2.0).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, np.ones(3))
    assert c.grad is None


def test_backward_on_non_grad_tensor_raises():
    with pytest.raises(RuntimeError):
        Tensor([1.0]).sum().backward()

