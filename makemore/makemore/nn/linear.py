import torch
from .model import Model
import math
from ._util import grad_add


class Linear(Model):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        generator: torch.Generator = torch.default_generator,
    ):
        self.w = torch.randn((in_feat, out_feat), generator=generator) / (
            math.sqrt(in_feat)
        )
        self.b = torch.zeros(out_feat) if bias else None
        self.wgrad = None
        self.bgrad = None
        self.training = True
        self.x = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        self.out = x @ self.w
        if self.b is not None:
            self.out += self.b
        return self.out

    # out_grad.shape = self.out.shape = (samples, out_feat)
    def backprop(self, out_grad: torch.Tensor) -> torch.Tensor:
        assert self.x is not None
        # grad
        with torch.no_grad():
            self.wgrad = grad_add(self.wgrad, self.x.T @ out_grad)
            if self.b is not None:
                self.bgrad = grad_add(self.bgrad, out_grad)
            xgrad = out_grad @ self.w.T

        return xgrad

    def parameters(self) -> list[torch.Tensor]:
        return [self.w] + ([] if self.b is None else [self.b])
