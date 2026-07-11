from .cross_entropy import cross_entropy, backprop_cross_entropy
import torch.nn.functional as F
import torch
import pytest


def test_cross_entropy():
    g = torch.Generator().manual_seed(1)
    for _ in range(100):
        size = 100
        a = torch.randn((size, 10), generator=g)
        ypred = torch.randint(0, 9, (size,), generator=g)
        ours = cross_entropy(a, ypred)
        others = F.cross_entropy(a, ypred)
        assert ours.item() == pytest.approx(others.item())


def test_grad():
    g = torch.Generator().manual_seed(3)
    for i in range(10):
        size = 100
        a = torch.randn((size, 10), generator=g, requires_grad=True)
        ypred = torch.randint(0, 9, (size,), generator=g)
        out = cross_entropy(a, ypred)
        out.backward()
        grad = backprop_cross_entropy(a, ypred, out, torch.ones_like(out))
        assert torch.allclose(grad, a.grad)
