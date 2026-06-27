import torch
from .model import Model

class Linear(Model):
    def __init__(self, in_feat: int, out_feat: int, bias: bool = True):
        self.w = torch.randn((in_feat, out_feat)) / in_feat**0.5
        self.b = torch.randn(out_feat) if bias else None
        self.training = True

    def __call__(self, x: torch.Tensor):
        return x * self.w + self.b

    def parameters(self) -> list[torch.Tensor]:
        return [self.w] + ([] if self.b is None else [self.b])
