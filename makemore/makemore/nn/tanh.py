from .model import Model
import torch


class Tanh(Model):
    def __init__(self):
        self.training = True

    def __call__(self, x: torch.Tensor):
        y = (2 * x).exp()
        self.out = (y - 1) / (y + 1)
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []
