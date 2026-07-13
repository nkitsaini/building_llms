from .linear import *
import torch


def test_grad():
    g = torch.Generator().manual_seed(3)
    for i in range(10):
        input = torch.rand(1, 20)
        layer1 = Linear(20, 10)
        layer2 = Linear(10, 1)
        for p in [*layer1.parameters(), *layer2.parameters()]:
            p.requires_grad = True
        out = layer2(layer1(input))
        out.backward()
        l2back = layer2.backprop(torch.ones(1, 1))
        layer1.backprop(l2back)
        assert torch.allclose(layer2.wgrad, layer2.w.grad)  # type: ignore
        assert torch.allclose(layer2.bgrad, layer2.b.grad)  # type: ignore
        assert torch.allclose(layer1.wgrad, layer1.w.grad)  # type: ignore
        assert torch.allclose(layer1.bgrad, layer1.b.grad)  # type: ignore
