import torch
from .model import Model

class Linear(Model):
    def __init__(self, in_feat: int, out_feat: int, bias: bool = True, generator: torch.Generator = torch.default_generator):
        self.w = torch.randn((in_feat, out_feat), generator=generator) / in_feat**0.5
        self.b = torch.randn(out_feat, generator=generator) if bias else None
        self.training = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out= (x @ self.w) + self.b
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.w] + ([] if self.b is None else [self.b])
